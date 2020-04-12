#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Cameras_Common_command_line_helper.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_rotation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_translation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/sfm_global_engine_relative_motions.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_report.hpp"
#include "openMVG/system/timer.hpp"
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres_camera_functor.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/types.hpp"
#include "/user/HS229/mt00853/Documents/Codes/SFM/Scratch/ceres-solver/include/ceres/problem.h"
#include "/user/HS229/mt00853/Documents/Codes/SFM/Scratch/ceres-solver/include/ceres/solver.h"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"

#include "/user/HS229/mt00853/Documents/Codes/SFM/Scratch/ceres-solver/include/ceres/rotation.h"
#include "/user/HS229/mt00853/Documents/Codes/SFM/Scratch/ceres-solver/include/ceres/types.h"

#include <iostream>
#include <limits>

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "sfm.hpp"
#include <cstdlib>

using namespace std;
using namespace openMVG;
using namespace openMVG::sfm;

ceres::CostFunction * IntrinsicsToCostFunction
        (
                openMVG::cameras::IntrinsicBase * intrinsic,
                const Vec2 & observation
        )
{
    const double weight = 0;
    switch (intrinsic->getType())
    {
        case openMVG::cameras::PINHOLE_CAMERA:
            return ResidualErrorFunctor_Pinhole_Intrinsic::Create(observation, weight);
        case openMVG::cameras::PINHOLE_CAMERA_RADIAL1:
            return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1::Create(observation, weight);
        case openMVG::cameras::PINHOLE_CAMERA_RADIAL3:
            return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(observation, weight);
        case openMVG::cameras::PINHOLE_CAMERA_BROWN:
            return ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2::Create(observation, weight);
        case openMVG::cameras::PINHOLE_CAMERA_FISHEYE:
            return ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye::Create(observation, weight);
        case openMVG::cameras::CAMERA_SPHERICAL:
            return ResidualErrorFunctor_Intrinsic_Spherical::Create(intrinsic, observation, weight);
        default:
            return {};
    }
}

int final_BA(std::string sSfM_Data_Filename,
             std::vector <int>& row_vec, std::vector <int>& col_vec, std::vector <double>& grad_vec)
{
    SfM_Data sfm_data;
    if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS|EXTRINSICS|STRUCTURE))) {return 0;}

    ceres::Problem problema;

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
    for (const auto & intrinsic_it : sfm_data.intrinsics)
    {
        const IndexT indexCam = intrinsic_it.first;
        map_intrinsics[indexCam] = intrinsic_it.second->getParams();
    }


    for (const auto & pose_it : map_poses)
    {
        const IndexT indexPose = pose_it.first;

        double * parameter_block = &map_poses.at(indexPose)[0];
        problema.AddParameterBlock(parameter_block, 6);
    }
    for (const auto & intrinsic_it : map_intrinsics)
    {
        const IndexT indexCam = intrinsic_it.first;
        double * parameter_block = &map_intrinsics.at(indexCam)[0];
        problema.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
    }

    // Set a LossFunction to be less penalized by false measurements
    // - set it to nullptr if you don't want use a lossFunction.
    ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(Square(4.0));

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
            ceres::CostFunction* cost_function = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(obs_it.second.x);

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
    }

    // Configure a BA engine and run it
    //  Make Ceres automatically detect the bundle structure.
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.max_num_iterations = 500;
    ceres_config_options.preconditioner_type =
            static_cast<ceres::PreconditionerType>(ceres::JACOBI);
    ceres_config_options.linear_solver_type =
            static_cast<ceres::LinearSolverType>(ceres::DENSE_SCHUR);
    ceres_config_options.sparse_linear_algebra_library_type =
            static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres::SUITE_SPARSE);
    ceres_config_options.minimizer_progress_to_stdout = true;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
    ceres_config_options.parameter_tolerance = 1e-8;

    // Solve BA
    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    //TODO Extract Jacobian - additional lines
    ceres::CRSMatrix jacob;

    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jacob);
    std::cout << "Jacobian size rows: " << jacob.num_rows << "\n";
    std::cout << "Jacobian size cols: " << jacob.num_cols << "\n";
    std::cout << "Jacobian size rows: " << jacob.values.size() << "\n";

    row_vec = jacob.rows;
    col_vec = jacob.cols;
    grad_vec = jacob.values;

    std::cout << col_vec.size() << " and " << jacob.cols.size() << "\n";
    std::cout << "Outputting the 100th vector element: " << jacob.rows[100] << "\n";
    std::cout << "Outputting the 100th vector element: " << jacob.cols[100] << "\n";
    std::cout << "Outputting the 100th vector element: " << jacob.values[100] << "\n";
    //end new lines
    // If no error, get back refined parameters
    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
        return false;
    }
    else // Solution is usable
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


int main()
{
    std::vector <int> row_vec;
    std::vector <int> col_vec;
    std::vector <double> grad_vec;

    std::string sSfM_Data_Filename = "/home/matteo/SFM/experiments/results/matches/sfm_data.json";
    int out = 0;
    std::cout << "if you read this, maybe not everything crashes";
    out = final_BA(sSfM_Data_Filename, row_vec, col_vec, grad_vec);
    std::cout << "\n\n\n\n\n" << out << "\n\n\n\n\n";
    return 0;
}