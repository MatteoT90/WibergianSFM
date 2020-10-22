// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_BA_ceres.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include "ceres/problem.h"
#include "ceres/solver.h"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"
//- Robust estimation - LMeds (since no threshold can be defined)
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres_camera_functor.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/types.hpp"
#include "sfm.hpp"

#include <ceres/rotation.h>
#include <ceres/types.h>

#include <iostream>
#include <limits>

namespace openMVG {
namespace sfm {

using namespace openMVG::cameras;
using namespace openMVG::geometry;

void extract_rotation
(
    const Mat3 R,
    double* angleAxis
)
{
  ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
  return;
}

// Ceres CostFunctor used for SfM pose center to GPS pose center minimization
struct PoseCenterConstraintCostFunction
{
  Vec3 weight_;
  Vec3 pose_center_constraint_;

  PoseCenterConstraintCostFunction
  (
    const Vec3 & center,
    const Vec3 & weight
  ): weight_(weight), pose_center_constraint_(center)
  {
  }

  template <typename T> bool
  operator()
  (
    const T* const cam_extrinsics, // R_t
    T* residuals
  )
  const
  {
    using Vec3T = Eigen::Matrix<T,3,1>;
    Eigen::Map<const Vec3T> cam_R(&cam_extrinsics[0]);
    Eigen::Map<const Vec3T> cam_t(&cam_extrinsics[3]);
    const Vec3T cam_R_transpose(-cam_R);

    Vec3T pose_center;
    // Rotate the point according the camera rotation
    ceres::AngleAxisRotatePoint(cam_R_transpose.data(), cam_t.data(), pose_center.data());
    pose_center = pose_center * T(-1);

    Eigen::Map<Vec3T> residuals_eigen(residuals);
    residuals_eigen = weight_.cast<T>().cwiseProduct(pose_center - pose_center_constraint_.cast<T>());

    return true;
  }
};

/// Create the appropriate cost functor according the provided input camera intrinsic model.
/// The residual can be weighetd if desired (default 0.0 means no weight).
ceres::CostFunction * IntrinsicsToCostFunction
(
  IntrinsicBase * intrinsic,
  const Vec2 & observation,
  const double weight
)
{
  switch (intrinsic->getType())
  {
    case PINHOLE_CAMERA:
      return ResidualErrorFunctor_Pinhole_Intrinsic::Create(observation, weight);
    case PINHOLE_CAMERA_RADIAL1:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1::Create(observation, weight);
    case PINHOLE_CAMERA_RADIAL3:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(observation, weight);
    case PINHOLE_CAMERA_BROWN:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2::Create(observation, weight);
    case PINHOLE_CAMERA_FISHEYE:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye::Create(observation, weight);
    case CAMERA_SPHERICAL:
      return ResidualErrorFunctor_Intrinsic_Spherical::Create(intrinsic, observation, weight);
    default:
      return {};
  }
}

Bundle_Adjustment_Ceres::BA_Ceres_options::BA_Ceres_options
(
  const bool bVerbose,
  bool bmultithreaded
)
: bVerbose_(bVerbose),
  nb_threads_(1),
  parameter_tolerance_(1e-8), //~= numeric_limits<float>::epsilon()
  bUse_loss_function_(true)
{
  #ifdef OPENMVG_USE_OPENMP
    nb_threads_ = omp_get_max_threads();
  #endif // OPENMVG_USE_OPENMP
  if (!bmultithreaded)
    nb_threads_ = 1;

  bCeres_summary_ = false;

  // Default configuration use a DENSE representation
  linear_solver_type_ = ceres::DENSE_SCHUR;
  preconditioner_type_ = ceres::JACOBI;
  // If Sparse linear solver are available
  // Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
  if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
  {
    sparse_linear_algebra_library_type_ = ceres::SUITE_SPARSE;
    linear_solver_type_ = ceres::SPARSE_SCHUR;
  }
  else
  {
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
    {
      sparse_linear_algebra_library_type_ = ceres::CX_SPARSE;
      linear_solver_type_ = ceres::SPARSE_SCHUR;
    }
    else
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
      sparse_linear_algebra_library_type_ = ceres::EIGEN_SPARSE;
      linear_solver_type_ = ceres::SPARSE_SCHUR;
    }
  }
}


Bundle_Adjustment_Ceres::Bundle_Adjustment_Ceres
(
  const Bundle_Adjustment_Ceres::BA_Ceres_options & options
)
: ceres_options_(options)
{}

Bundle_Adjustment_Ceres::BA_Ceres_options &
Bundle_Adjustment_Ceres::ceres_options()
{
  return ceres_options_;
}

bool Bundle_Adjustment_Ceres::Adjust
(
  SfM_Data & sfm_data,     // the SfM scene to refine
  int* vec_rows, int g,
  int* vec_cols, int h,
  double* vec_grad, int i
)
{
  //----------
  // Add camera parameters
  // - intrinsics
  // - poses [R|t]

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  //----------

  ceres::Problem problem;

  // Data wrapper for refinement:
  Hash_Map<IndexT, std::vector<double>> map_intrinsics;
  Hash_Map<IndexT, std::vector<double>> map_poses;

  // Setup Poses data & subparametrization
  for (const auto & pose_it : sfm_data.poses)
  {
    const IndexT indexPose = pose_it.first;

    const Pose3 & pose = pose_it.second;
    const Mat3 R = pose.rotation();
    const Vec3 t = pose.translation();

    double angleAxis[3];
    ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
    // angleAxis + translation
    map_poses[indexPose] = {angleAxis[0], angleAxis[1], angleAxis[2], t(0), t(1), t(2)};
  }

  // Setup Intrinsics data & subparametrization
  int iii = 0;
  for (const auto & intrinsic_it : sfm_data.intrinsics)
  {
    iii++;
    std::cout << iii << "\n";
    const IndexT indexCam = intrinsic_it.first;
    map_intrinsics[indexCam] = intrinsic_it.second->getParams();
  }

  for (const auto & pose_it : map_poses)
  {
        const IndexT indexPose = pose_it.first;

        double * parameter_block = &map_poses.at(indexPose)[0];
        problem.AddParameterBlock(parameter_block, 6);
  }
  for (const auto & intrinsic_it : map_intrinsics)
  {
    const IndexT indexCam = intrinsic_it.first;
    double * parameter_block = &map_intrinsics.at(indexCam)[0];
    problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
  }

  // std::cout << "Size of map_intrinsic: " << map_intrinsics.size() << " Size of map_poses: " << map_poses.size() << "\n";
  // std::cout << map_intrinsics.at(0).size() << " " << map_intrinsics.at(0)[1] << " " << map_intrinsics.at(0)[2] << "\n";
  // std::cout << map_poses.at(0).size() << " " << map_poses.at(0)[1] << " " << map_poses.at(0)[5] << "\n";
  // std::cout << map_poses.at(4).size() << " " << map_poses.at(4)[1] << " " << map_poses.at(4)[5] << "\n";
  // std::cout << map_poses.at(8).size() << " " << map_poses.at(8)[1] << " " << map_poses.at(8)[5] << "\n";
  // Set a LossFunction to be less penalized by false measurements
  // - set it to nullptr if you don't want use a lossFunction.
  ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(Square(4.0));

    // For all visibility add reprojections errors:
  int aaa = 0;
  for (auto & structure_landmark_it : sfm_data.structure)
  {
    const Observations & obs = structure_landmark_it.second.obs;
    for (const auto & obs_it : obs)
    {
      // Build the residual block corresponding to the track observation:
      const View * view = sfm_data.views.at(obs_it.first).get();

      // Each Residual block takes a point and a camera as input and outputs a 2
      // dimensional residual. Internally, the cost function stores the observed
      // image location and compares the reprojection against the observation.
      // the type is always 3
      // std::cout << sfm_data.intrinsics.at(view->id_intrinsic).get()-> getType() << "\n";
      ceres::CostFunction* cost_function =
        IntrinsicsToCostFunction(sfm_data.intrinsics.at(view->id_intrinsic).get(),
                                 obs_it.second.x);
      //ceres::CostFunction* cost_function = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(obs_it.second.x);
      if (cost_function)
      {
        problem.AddResidualBlock(cost_function,
          p_LossFunction,
          &map_intrinsics.at(0)[0],
          &map_poses.at(view->id_pose)[0],
          structure_landmark_it.second.X.data());
        // &map_intrinsics.at(view->id_intrinsic)[0],
        aaa++;
      }
      else
      {
        std::cerr << "Cannot create a CostFunction for this camera model." << std::endl;
        return false;
      }
    }
  }

  std::cout << "Ceres problem sizes :: " << problem.NumParameterBlocks() << " " << problem.NumParameters() << " " << problem.NumResidualBlocks() << " " << problem.NumResiduals() << "\n";

  std::cout << "Size of problem: " << aaa << "\n";
  // Configure a BA engine and run it
  //  Make Ceres automatically detect the bundle structure.
  ceres::Solver::Options ceres_config_options;
  ceres_config_options.max_num_iterations = 500;
  ceres_config_options.preconditioner_type =
    static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
  ceres_config_options.linear_solver_type =
    static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
  ceres_config_options.sparse_linear_algebra_library_type =
    static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
  ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
  ceres_config_options.logging_type = ceres::SILENT;
  ceres_config_options.num_threads = ceres_options_.nb_threads_;
  #if CERES_VERSION_MAJOR < 2
  ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
  #endif
  ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

  // Solve BA
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_config_options, &problem, &summary);
  if (ceres_options_.bCeres_summary_)
    std::cout << summary.FullReport() << std::endl;

  //TODO Extract Jacobian - additional lines
  ceres::CRSMatrix jacob;

  problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jacob);

  std::cout << "Jacobian size rows: " << jacob.num_rows << " " << jacob.rows.size() << "\n";
  std::cout << "Jacobian size cols: " << jacob.num_cols << " " << jacob.cols.size() << "\n";
  std::cout << "Jacobian size rows: " << jacob.values.size() << "\n";

  vec_rows[0] = jacob.rows.size();
  vec_cols[0] = jacob.cols.size();
  vec_grad[0] = jacob.values.size();

  for(int ii=0; ii<jacob.rows.size(); ii++)  {vec_rows[1+ii] = jacob.rows[ii];}
  for(int ii=0; ii<jacob.cols.size(); ii++)  {vec_cols[1+ii] = jacob.cols[ii];}
  for(int ii=0; ii<jacob.values.size(); ii++){vec_grad[1+ii] = jacob.values[ii];}

  std::cout << "Bundle Adjustment reached this point." << std::endl;
  //end new lines
  // If no error, get back refined parameters
  if (!summary.IsSolutionUsable())
  {
    if (ceres_options_.bVerbose_)
      std::cout << "Bundle Adjustment failed." << std::endl;
    return false;
  }
  else // Solution is usable
  {
    if (ceres_options_.bVerbose_)
    {
      // Display statistics about the minimization
      std::cout << std::endl
        << "Bundle Adjustment statistics (approximated RMSE):\n"
        << " #views: " << sfm_data.views.size() << "\n"
        << " #poses: " << sfm_data.poses.size() << "\n"
        << " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
        << " #tracks: " << sfm_data.structure.size() << "\n"
        << " #residuals: " << summary.num_residuals << "\n"
        << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
        << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
        << " Time (s): " << summary.total_time_in_seconds << "\n"
        << std::endl;
    }

    // Update camera poses with refined data
        for (auto & pose_it : sfm_data.poses)
      {
        const IndexT indexPose = pose_it.first;

        Mat3 R_refined;
        ceres::AngleAxisToRotationMatrix(&map_poses.at(indexPose)[0], R_refined.data());
        Vec3 t_refined(map_poses.at(indexPose)[3], map_poses.at(indexPose)[4], map_poses.at(indexPose)[5]);
        // Update the pose
        Pose3 & pose = pose_it.second;
        pose = Pose3(R_refined, -R_refined.transpose() * t_refined);
      }

    // Update camera intrinsics with refined data
      for (auto & intrinsic_it : sfm_data.intrinsics)
      {
        const IndexT indexCam = intrinsic_it.first;

        const std::vector<double> & vec_params = map_intrinsics.at(indexCam);
        intrinsic_it.second->updateFromParams(vec_params);
      }
    return true;
  }
}

bool Bundle_Adjustment_Ceres::GlobalAdjust
(
        double* pose, int l_pose,
        double* intrinsics, int l_int,
        double* cloud, int l_cloud,
        double* features, int l_feat,
        int* cams, int l_cam,
        int* track, int l_track,
        double* weight, int l_w,
        int* vec_rows, int l_vr,
        int* vec_cols, int l_vc,
        double* vec_grad, int l_vg,
        double* residuals_out, int l_res,
        double* gradient_out, int l_grad,
        int optimise
)
{
// Data wrapper for refinement:
Hash_Map<IndexT, std::vector<double>> map_intrinsics;
Hash_Map<IndexT, std::vector<double>> map_poses;
Hash_Map<IndexT, Vec3> map_cloud;
Hash_Map<IndexT, Vec2> map_obs;

// Setup Poses data & subparametrization
for (int ii = 0; ii < l_pose / 6; ii++){ map_poses[ii] = {pose[ii * 6], pose[ii * 6 + 1], pose[ii * 6 + 2],
                                                          pose[ii * 6 + 3], pose[ii * 6 + 4], pose[ii * 6 + 5]};}

for (int ii = 0; ii < l_int / 6; ii++){ map_intrinsics[ii] = {intrinsics[ii * 6], intrinsics[1 + ii * 6],
                                                              intrinsics[2+ii*6], intrinsics[3+ii*6],
                                                              intrinsics[4+ii*6], intrinsics[5+ii*6]};}

for(int ll = 0; ll < l_cloud / 3; ll++)
        {map_cloud[ll] = (Vec3 () << cloud[3 * ll], cloud[3 * ll + 1], cloud[3 * ll + 2]).finished();}
for(int ll = 0; ll < l_feat / 2; ll++)
        {map_obs[ll] = (Vec2 () << features[2 * ll], features[2 * ll + 1]).finished();}

if(optimise != 0){
    for(int mode; mode < 3; mode++)
    {
        ceres::Solver::Summary summary;

        //ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(Square(4.0));
        ceres::Solver::Options ceres_config_options;
        ceres_config_options.max_num_iterations = 500;
        ceres_config_options.preconditioner_type =
                static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
        ceres_config_options.linear_solver_type =
        static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);

        // ceres_config_options.minimizer_type = ceres::TRUST_REGION;
        // ceres_config_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

        ceres_config_options.sparse_linear_algebra_library_type =
                static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
        ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
        ceres_config_options.logging_type = ceres::SILENT;
        ceres_config_options.num_threads = ceres_options_.nb_threads_;
        #if CERES_VERSION_MAJOR < 2
        ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
        #endif
        ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

        ceres::Problem problem;

        // std::cout << "Ceres problem sizes :: " << problem.NumParameterBlocks() << " " << problem.NumParameters()
        // << " " << problem.NumResidualBlocks() << " " << problem.NumResiduals() << "\n";

        for (const auto & pose_it : map_poses)
        {
            const IndexT indexPose = pose_it.first;
            double * parameter_block = &map_poses.at(indexPose)[0];
            problem.AddParameterBlock(parameter_block, 6);
            if(mode == 0)
            {
                std::vector<int> keep_constant;
                keep_constant.insert(keep_constant.end(), {0,1,2});
                ceres::SubsetParameterization *subset_parameterization = new ceres::SubsetParameterization(6, keep_constant);
                problem.SetParameterization(parameter_block, subset_parameterization);
            }
        }

        for (const auto & intrinsic_it : map_intrinsics)
        {
            const IndexT indexCam = intrinsic_it.first;
            double * parameter_block = &map_intrinsics.at(indexCam)[0];
            problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
            if(mode < 2)
            {
                problem.SetParameterBlockConstant(parameter_block);
            }
        }

        for(int ll = 0; ll < l_feat / 2; ll++)
        {
        ceres::CostFunction* cost_function = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(map_obs[ll], weight[ll]);
        //problem.AddResidualBlock(cost_function,
        //                     p_LossFunction,
        //                     &map_intrinsics.at(0)[0],
        //                     &map_poses.at(cams[ll])[0],
        //                     &map_cloud.at(track[ll])[0]);
        //}

	problem.AddResidualBlock(cost_function,
				 NULL,
				 &map_intrinsics.at(0)[0],
				 &map_poses.at(cams[ll])[0],
				 &map_cloud.at(track[ll])[0]);
	}
    
	
        ceres::Solve(ceres_config_options, &problem, &summary);
        //std::cout << std::endl
        //    << "Bundle Adjustment statistics (approximated RMSE):\n"
        //    << " #residuals: " << summary.num_residuals << "\n"
        //    << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
        //    << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
        //    << " Time (s): " << summary.total_time_in_seconds << "\n"
        //    << std::endl;
    }

    for (int ii = 0; ii < l_pose / 6; ii++){
        for(int dc = 0; dc < 6; dc++){pose[ii * 6 + dc] = map_poses.at(ii)[dc];}
    }
    for (int ii = 0; ii < l_int / 6; ii++){
        for(int dc = 0; dc < 6; dc++){ intrinsics[ii * 6 + dc] = map_intrinsics.at(ii)[dc];}
    }
    for (int ii = 0; ii < l_cloud / 3; ii++){
        for(int dc = 0; dc < 3; dc++){ cloud[ii * 3 + dc] = map_cloud.at(ii)[dc];}
    }

}

// std::cout << "Running just the gradient evaluation part \n";

ceres::Solver::Summary summary;
ceres::CRSMatrix jacob;
double cost = 0;
std::vector <double> residuals;
std::vector <double> gradients;

ceres::Problem eval_problem;

//ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(Square(4.0));
ceres::Solver::Options ceres_config_options;
ceres_config_options.max_num_iterations = 500;
ceres_config_options.preconditioner_type =
        static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
ceres_config_options.linear_solver_type =
static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
ceres_config_options.sparse_linear_algebra_library_type =
        static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
ceres_config_options.logging_type = ceres::SILENT;
ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

for (const auto & pose_it : map_poses)
{
    const IndexT indexPose = pose_it.first;
    double * parameter_block = &map_poses.at(indexPose)[0];
    eval_problem.AddParameterBlock(parameter_block, 6);
}
for (const auto & intrinsic_it : map_intrinsics)
{
    const IndexT indexCam = intrinsic_it.first;
    double * parameter_block = &map_intrinsics.at(indexCam)[0];
    eval_problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
}
for(int ll = 0; ll < l_feat / 2; ll++)
{
ceres::CostFunction* cost_function = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(map_obs[ll], weight[ll]);
//eval_problem.AddResidualBlock(cost_function,
//                              p_LossFunction,
//                              &map_intrinsics.at(0)[0],
//                              &map_poses.at(cams[ll])[0],
//                              &map_cloud.at(track[ll])[0]);
//}

eval_problem.AddResidualBlock(cost_function,
			       NULL,
			       &map_intrinsics.at(0)[0],
			       &map_poses.at(cams[ll])[0],
			       &map_cloud.at(track[ll])[0]);
}
  
 
// problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jacob);
eval_problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &residuals, &gradients, &jacob);

for(int ii=0; ii<jacob.rows.size(); ii++)  {vec_rows[2+ii] = jacob.rows[ii];}
for(int ii=0; ii<jacob.cols.size(); ii++)  {vec_cols[2+ii] = jacob.cols[ii];}
for(int ii=0; ii<jacob.values.size(); ii++){vec_grad[1+ii] = jacob.values[ii];}
for(int ii=0; ii<residuals.size(); ii++){residuals_out[1+ii] = residuals[ii];}
for(int ii=0; ii<gradients.size(); ii++){gradient_out[1+ii] = gradients[ii];}

vec_rows[0] = jacob.rows.size();
vec_cols[0] = jacob.cols.size(); vec_grad[0] = jacob.values.size();
vec_rows[1] = jacob.num_rows; vec_cols[1] = jacob.num_cols;
residuals_out[0] = residuals.size();
gradient_out[0] = gradients.size();

return true;
}

bool Bundle_Adjustment_Ceres::GlobalAdjustHuber
        (
                double* pose, int l_pose,
                double* intrinsics, int l_int,
                double* cloud, int l_cloud,
                double* features, int l_feat,
                int* cams, int l_cam,
                int* track, int l_track,
                double* weight, int l_w,
                int* vec_rows, int l_vr,
                int* vec_cols, int l_vc,
                double* vec_grad, int l_vg,
                double* residuals_out, int l_res,
                double* gradient_out, int l_grad,
                int optimise
        )
{
// Data wrapper for refinement:
    Hash_Map<IndexT, std::vector<double>> map_intrinsics;
    Hash_Map<IndexT, std::vector<double>> map_poses;
    Hash_Map<IndexT, Vec3> map_cloud;
    Hash_Map<IndexT, Vec2> map_obs;

// Setup Poses data & subparametrization
    for (int ii = 0; ii < l_pose / 6; ii++){ map_poses[ii] = {pose[ii * 6], pose[ii * 6 + 1], pose[ii * 6 + 2],
                                                              pose[ii * 6 + 3], pose[ii * 6 + 4], pose[ii * 6 + 5]};}

    for (int ii = 0; ii < l_int / 6; ii++){ map_intrinsics[ii] = {intrinsics[ii * 6], intrinsics[1 + ii * 6],
                                                                  intrinsics[2+ii*6], intrinsics[3+ii*6],
                                                                  intrinsics[4+ii*6], intrinsics[5+ii*6]};}

    for(int ll = 0; ll < l_cloud / 3; ll++)
    {map_cloud[ll] = (Vec3 () << cloud[3 * ll], cloud[3 * ll + 1], cloud[3 * ll + 2]).finished();}
    for(int ll = 0; ll < l_feat / 2; ll++)
    {map_obs[ll] = (Vec2 () << features[2 * ll], features[2 * ll + 1]).finished();}

    if(optimise != 0){
        for(int mode; mode < 3; mode++)
        {
            ceres::Solver::Summary summary;

            ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(Square(4.0));
            ceres::Solver::Options ceres_config_options;
            ceres_config_options.max_num_iterations = 500;
            ceres_config_options.preconditioner_type =
                    static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
            ceres_config_options.linear_solver_type =
                    static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);

            // ceres_config_options.minimizer_type = ceres::TRUST_REGION;
            // ceres_config_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

            ceres_config_options.sparse_linear_algebra_library_type =
                    static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
            ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
            ceres_config_options.logging_type = ceres::SILENT;
            ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
            ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
            ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

            ceres::Problem problem;

            // std::cout << "Ceres problem sizes :: " << problem.NumParameterBlocks() << " " << problem.NumParameters()
            // << " " << problem.NumResidualBlocks() << " " << problem.NumResiduals() << "\n";

            for (const auto & pose_it : map_poses)
            {
                const IndexT indexPose = pose_it.first;
                double * parameter_block = &map_poses.at(indexPose)[0];
                problem.AddParameterBlock(parameter_block, 6);
                if(mode == 0)
                {
                    std::vector<int> keep_constant;
                    keep_constant.insert(keep_constant.end(), {0,1,2});
                    ceres::SubsetParameterization *subset_parameterization = new ceres::SubsetParameterization(6, keep_constant);
                    problem.SetParameterization(parameter_block, subset_parameterization);
                }
            }

            for (const auto & intrinsic_it : map_intrinsics)
            {
                const IndexT indexCam = intrinsic_it.first;
                double * parameter_block = &map_intrinsics.at(indexCam)[0];
                problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
                if(mode < 2)
                {
                    problem.SetParameterBlockConstant(parameter_block);
                }
            }

            for(int ll = 0; ll < l_feat / 2; ll++)
            {
                ceres::CostFunction* cost_function = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(map_obs[ll], weight[ll]);
                problem.AddResidualBlock(cost_function,
                                     p_LossFunction,
                                     &map_intrinsics.at(0)[0],
                                     &map_poses.at(cams[ll])[0],
                                     &map_cloud.at(track[ll])[0]);
                }

                //problem.AddResidualBlock(cost_function,
                //                         NULL,
                //                         &map_intrinsics.at(0)[0],
                //                         &map_poses.at(cams[ll])[0],
                //                         &map_cloud.at(track[ll])[0]);
                //}


            ceres::Solve(ceres_config_options, &problem, &summary);
            //std::cout << std::endl
            //    << "Bundle Adjustment statistics (approximated RMSE):\n"
            //    << " #residuals: " << summary.num_residuals << "\n"
            //    << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
            //    << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
            //    << " Time (s): " << summary.total_time_in_seconds << "\n"
            //    << std::endl;
        }

        for (int ii = 0; ii < l_pose / 6; ii++){
            for(int dc = 0; dc < 6; dc++){pose[ii * 6 + dc] = map_poses.at(ii)[dc];}
        }
        for (int ii = 0; ii < l_int / 6; ii++){
            for(int dc = 0; dc < 6; dc++){ intrinsics[ii * 6 + dc] = map_intrinsics.at(ii)[dc];}
        }
        for (int ii = 0; ii < l_cloud / 3; ii++){
            for(int dc = 0; dc < 3; dc++){ cloud[ii * 3 + dc] = map_cloud.at(ii)[dc];}
        }

    }

// std::cout << "Running just the gradient evaluation part \n";

    ceres::Solver::Summary summary;
    ceres::CRSMatrix jacob;
    double cost = 0;
    std::vector <double> residuals;
    std::vector <double> gradients;

    ceres::Problem eval_problem;

    ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(Square(4.0));
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.max_num_iterations = 500;
    ceres_config_options.preconditioner_type =
            static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
    ceres_config_options.linear_solver_type =
            static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
    ceres_config_options.sparse_linear_algebra_library_type =
            static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
    ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
    ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
    ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

    for (const auto & pose_it : map_poses)
    {
        const IndexT indexPose = pose_it.first;
        double * parameter_block = &map_poses.at(indexPose)[0];
        eval_problem.AddParameterBlock(parameter_block, 6);
    }
    for (const auto & intrinsic_it : map_intrinsics)
    {
        const IndexT indexCam = intrinsic_it.first;
        double * parameter_block = &map_intrinsics.at(indexCam)[0];
        eval_problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
    }
    for(int ll = 0; ll < l_feat / 2; ll++)
    {
        ceres::CostFunction* cost_function = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(map_obs[ll], weight[ll]);
        eval_problem.AddResidualBlock(cost_function,
        p_LossFunction,
        &map_intrinsics.at(0)[0],
        &map_poses.at(cams[ll])[0],
        &map_cloud.at(track[ll])[0]);
    }

        //eval_problem.AddResidualBlock(cost_function,
        //                              NULL,
        //                              &map_intrinsics.at(0)[0],
        //                              &map_poses.at(cams[ll])[0],
        //                              &map_cloud.at(track[ll])[0]);
        //}


    // problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jacob);
    eval_problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &residuals, &gradients, &jacob);

    for(int ii=0; ii<jacob.rows.size(); ii++)  {vec_rows[2+ii] = jacob.rows[ii];}
    for(int ii=0; ii<jacob.cols.size(); ii++)  {vec_cols[2+ii] = jacob.cols[ii];}
    for(int ii=0; ii<jacob.values.size(); ii++){vec_grad[1+ii] = jacob.values[ii];}
    for(int ii=0; ii<residuals.size(); ii++){residuals_out[1+ii] = residuals[ii];}
    for(int ii=0; ii<gradients.size(); ii++){gradient_out[1+ii] = gradients[ii];}

    vec_rows[0] = jacob.rows.size();
    vec_cols[0] = jacob.cols.size(); vec_grad[0] = jacob.values.size();
    vec_rows[1] = jacob.num_rows; vec_cols[1] = jacob.num_cols;
    residuals_out[0] = residuals.size();
    gradient_out[0] = gradients.size();

    return true;
}

bool Bundle_Adjustment_Ceres::Adjust
(
        SfM_Data & sfm_data,     // the SfM scene to refine
        const Optimize_Options & options
)
{
//----------
// Add camera parameters
// - intrinsics
// - poses [R|t]

// Create residuals for each observation in the bundle adjustment problem. The
// parameters for cameras and points are added automatically.
//----------


double pose_center_robust_fitting_error = 0.0;
openMVG::geometry::Similarity3 sim_to_center;
bool b_usable_prior = false;
if (options.use_motion_priors_opt && sfm_data.GetViews().size() > 3)
{
// - Compute a robust X-Y affine transformation & apply it
// - This early transformation enhance the conditionning (solution closer to the Prior coordinate system)
{
    // Collect corresponding camera centers
    std::vector<Vec3> X_SfM, X_GPS;
    for (const auto & view_it : sfm_data.GetViews())
    {
        const sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
            X_SfM.push_back( sfm_data.GetPoses().at(prior->id_pose).center() );
            X_GPS.push_back( prior->pose_center_ );
        }
    }
    openMVG::geometry::Similarity3 sim;

    // Compute the registration:
    if (X_GPS.size() > 3)
    {
        const Mat X_SfM_Mat = Eigen::Map<Mat>(X_SfM[0].data(),3, X_SfM.size());
        const Mat X_GPS_Mat = Eigen::Map<Mat>(X_GPS[0].data(),3, X_GPS.size());
        geometry::kernel::Similarity3_Kernel kernel(X_SfM_Mat, X_GPS_Mat);
        const double lmeds_median = openMVG::robust::LeastMedianOfSquares(kernel, &sim);
        if (lmeds_median != std::numeric_limits<double>::max())
        {
            b_usable_prior = true; // PRIOR can be used safely

            // Compute the median residual error once the registration is applied
            for (Vec3 & pos : X_SfM) // Transform SfM poses for residual computation
            {
                pos = sim(pos);
            }
            Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
            std::sort(residual.data(), residual.data() + residual.size());
            pose_center_robust_fitting_error = residual(residual.size()/2);

            // Apply the found transformation to the SfM Data Scene
            openMVG::sfm::ApplySimilarity(sim, sfm_data);

            // Move entire scene to center for better numerical stability
            Vec3 pose_centroid = Vec3::Zero();
            for (const auto & pose_it : sfm_data.poses)
            {
                pose_centroid += (pose_it.second.center() / (double)sfm_data.poses.size());
            }
            sim_to_center = openMVG::geometry::Similarity3(openMVG::sfm::Pose3(Mat3::Identity(), pose_centroid), 1.0);
            openMVG::sfm::ApplySimilarity(sim_to_center, sfm_data, true);
        }
    }
}
}

ceres::Problem problem;

// Data wrapper for refinement:
Hash_Map<IndexT, std::vector<double>> map_intrinsics;
Hash_Map<IndexT, std::vector<double>> map_poses;

// Setup Poses data & subparametrization
for (const auto & pose_it : sfm_data.poses)
{
const IndexT indexPose = pose_it.first;

const Pose3 & pose = pose_it.second;
const Mat3 R = pose.rotation();
const Vec3 t = pose.translation();

double angleAxis[3];
ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
// angleAxis + translation
map_poses[indexPose] = {angleAxis[0], angleAxis[1], angleAxis[2], t(0), t(1), t(2)};

double * parameter_block = &map_poses.at(indexPose)[0];
problem.AddParameterBlock(parameter_block, 6);
if (options.extrinsics_opt == Extrinsic_Parameter_Type::NONE)
{
    // set the whole parameter block as constant for best performance
    problem.SetParameterBlockConstant(parameter_block);
}
else  // Subset parametrization
{
    std::vector<int> vec_constant_extrinsic;
    // If we adjust only the translation, we must set ROTATION as constant
    if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_TRANSLATION)
    {
        // Subset rotation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {0,1,2});
    }
    // If we adjust only the rotation, we must set TRANSLATION as constant
    if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_ROTATION)
    {
        // Subset translation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {3,4,5});
    }
    if (!vec_constant_extrinsic.empty())
    {
        ceres::SubsetParameterization *subset_parameterization =
                new ceres::SubsetParameterization(6, vec_constant_extrinsic);
        problem.SetParameterization(parameter_block, subset_parameterization);
    }
}
}

// Setup Intrinsics data & subparametrization
for (const auto & intrinsic_it : sfm_data.intrinsics)
{
const IndexT indexCam = intrinsic_it.first;
if (isValid(intrinsic_it.second->getType()))
{
    map_intrinsics[indexCam] = intrinsic_it.second->getParams();
    if (!map_intrinsics.at(indexCam).empty())
    {
        double * parameter_block = &map_intrinsics.at(indexCam)[0];
        problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
        if (options.intrinsics_opt == Intrinsic_Parameter_Type::NONE)
        {
            // set the whole parameter block as constant for best performance
            problem.SetParameterBlockConstant(parameter_block);
        }
        else
        {
            const std::vector<int> vec_constant_intrinsic =
                    intrinsic_it.second->subsetParameterization(options.intrinsics_opt);
            if (!vec_constant_intrinsic.empty())
            {
                ceres::SubsetParameterization *subset_parameterization =
                        new ceres::SubsetParameterization(
                                map_intrinsics.at(indexCam).size(), vec_constant_intrinsic);
                problem.SetParameterization(parameter_block, subset_parameterization);
            }
        }
    }
}
else
{
    std::cerr << "Unsupported camera type." << std::endl;
}
}

// Set a LossFunction to be less penalized by false measurements
//  - set it to nullptr if you don't want use a lossFunction.
ceres::LossFunction * p_LossFunction =
    ceres_options_.bUse_loss_function_ ?
    new ceres::HuberLoss(Square(4.0))
                                       : nullptr;

// For all visibility add reprojections errors:
for (auto & structure_landmark_it : sfm_data.structure)
{
const Observations & obs = structure_landmark_it.second.obs;

for (const auto & obs_it : obs)
{
    // Build the residual block corresponding to the track observation:
    const View * view = sfm_data.views.at(obs_it.first).get();

    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    ceres::CostFunction* cost_function =
            IntrinsicsToCostFunction(sfm_data.intrinsics.at(view->id_intrinsic).get(),
                                     obs_it.second.x);

    if (cost_function)
    {
        if (!map_intrinsics.at(view->id_intrinsic).empty())
        {
            problem.AddResidualBlock(cost_function,
                                     p_LossFunction,
                                     &map_intrinsics.at(view->id_intrinsic)[0],
                                     &map_poses.at(view->id_pose)[0],
                                     structure_landmark_it.second.X.data());
        }
        else
        {
            problem.AddResidualBlock(cost_function,
                                     p_LossFunction,
                                     &map_poses.at(view->id_pose)[0],
                                     structure_landmark_it.second.X.data());
        }
    }
    else
    {
        std::cerr << "Cannot create a CostFunction for this camera model." << std::endl;
        return false;
    }
}
if (options.structure_opt == Structure_Parameter_Type::NONE)
    problem.SetParameterBlockConstant(structure_landmark_it.second.X.data());
}

if (options.control_point_opt.bUse_control_points)
{
// Use Ground Control Point:
// - fixed 3D points with weighted observations
for (auto & gcp_landmark_it : sfm_data.control_points)
{
    const Observations & obs = gcp_landmark_it.second.obs;

    for (const auto & obs_it : obs)
    {
        // Build the residual block corresponding to the track observation:
        const View * view = sfm_data.views.at(obs_it.first).get();

        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
                IntrinsicsToCostFunction(
                        sfm_data.intrinsics.at(view->id_intrinsic).get(),
                        obs_it.second.x,
                        options.control_point_opt.weight);

        if (cost_function)
        {
            if (!map_intrinsics.at(view->id_intrinsic).empty())
            {
                problem.AddResidualBlock(cost_function,
                                         nullptr,
                                         &map_intrinsics.at(view->id_intrinsic)[0],
                                         &map_poses.at(view->id_pose)[0],
                                         gcp_landmark_it.second.X.data());
            }
            else
            {
                problem.AddResidualBlock(cost_function,
                                         nullptr,
                                         &map_poses.at(view->id_pose)[0],
                                         gcp_landmark_it.second.X.data());
            }
        }
    }
    if (obs.empty())
    {
        std::cerr
                << "Cannot use this GCP id: " << gcp_landmark_it.first
                << ". There is not linked image observation." << std::endl;
    }
    else
    {
        // Set the 3D point as FIXED (it's a valid GCP)
        problem.SetParameterBlockConstant(gcp_landmark_it.second.X.data());
    }
}
}

// Add Pose prior constraints if any
if (b_usable_prior)
{
for (const auto & view_it : sfm_data.GetViews())
{
    const sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
    if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
    {
        // Add the cost functor (distance from Pose prior to the SfM_Data Pose center)
        ceres::CostFunction * cost_function =
                new ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 6>(
                        new PoseCenterConstraintCostFunction(prior->pose_center_, prior->center_weight_));

        problem.AddResidualBlock(
                cost_function,
                new ceres::HuberLoss(
                        Square(pose_center_robust_fitting_error)),
                &map_poses.at(prior->id_view)[0]);
    }
}
}

// Configure a BA engine and run it
//  Make Ceres automatically detect the bundle structure.
ceres::Solver::Options ceres_config_options;
ceres_config_options.max_num_iterations = 500;
ceres_config_options.preconditioner_type =
    static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
ceres_config_options.linear_solver_type =
    static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
ceres_config_options.sparse_linear_algebra_library_type =
    static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
ceres_config_options.logging_type = ceres::SILENT;
ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;

// Solve BA
ceres::Solver::Summary summary;
ceres::Solve(ceres_config_options, &problem, &summary);
if (ceres_options_.bCeres_summary_)
std::cout << summary.FullReport() << std::endl;

// If no error, get back refined parameters
if (!summary.IsSolutionUsable())
{
if (ceres_options_.bVerbose_)
    std::cout << "Bundle Adjustment failed." << std::endl;
return false;
}
else // Solution is usable
{
if (ceres_options_.bVerbose_)
{
    // Display statistics about the minimization
    std::cout << std::endl
              << "Bundle Adjustment statistics (approximated RMSE):\n"
              << " #views: " << sfm_data.views.size() << "\n"
              << " #poses: " << sfm_data.poses.size() << "\n"
              << " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
              << " #tracks: " << sfm_data.structure.size() << "\n"
              << " #residuals: " << summary.num_residuals << "\n"
              << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
              << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
              << " Time (s): " << summary.total_time_in_seconds << "\n"
              << std::endl;
    if (options.use_motion_priors_opt)
        std::cout << "Usable motion priors: " << (int)b_usable_prior << std::endl;
}

// Update camera poses with refined data
if (options.extrinsics_opt != Extrinsic_Parameter_Type::NONE)
{
    for (auto & pose_it : sfm_data.poses)
    {
        const IndexT indexPose = pose_it.first;

        Mat3 R_refined;
        ceres::AngleAxisToRotationMatrix(&map_poses.at(indexPose)[0], R_refined.data());
        Vec3 t_refined(map_poses.at(indexPose)[3], map_poses.at(indexPose)[4], map_poses.at(indexPose)[5]);
        // Update the pose
        Pose3 & pose = pose_it.second;
        pose = Pose3(R_refined, -R_refined.transpose() * t_refined);
    }
}

// Update camera intrinsics with refined data
if (options.intrinsics_opt != Intrinsic_Parameter_Type::NONE)
{
    for (auto & intrinsic_it : sfm_data.intrinsics)
    {
        const IndexT indexCam = intrinsic_it.first;

        const std::vector<double> & vec_params = map_intrinsics.at(indexCam);
        intrinsic_it.second->updateFromParams(vec_params);
    }
}

// Structure is already updated directly if needed (no data wrapping)

if (b_usable_prior)
{
    // set back to the original scene centroid
    openMVG::sfm::ApplySimilarity(sim_to_center.inverse(), sfm_data, true);

    //--
    // - Compute some fitting statistics
    //--

    // Collect corresponding camera centers
    std::vector<Vec3> X_SfM, X_GPS;
    for (const auto & view_it : sfm_data.GetViews())
    {
        const sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
            X_SfM.push_back( sfm_data.GetPoses().at(prior->id_pose).center() );
            X_GPS.push_back( prior->pose_center_ );
        }
    }
    // Compute the registration fitting error (once BA with Prior have been used):
    if (X_GPS.size() > 3)
    {
        // Compute the median residual error
        Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
        std::cout
                << "Pose prior statistics (user units):\n"
                << " - Starting median fitting error: " << pose_center_robust_fitting_error << "\n"
                << " - Final fitting error:";
        minMaxMeanMedian<Vec::Scalar>(residual.data(), residual.data() + residual.size());
    }
}
return true;
}
}

} // namespace sfm
} // namespace openMVG
