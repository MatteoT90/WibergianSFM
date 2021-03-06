#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Cameras_Common_command_line_helper.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_rotation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/GlobalSfM_translation_averaging.hpp"
#include "openMVG/sfm/pipelines/global/sfm_global_engine_relative_motions.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_report.hpp"
#include "openMVG/system/timer.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "sfm.hpp"
#include <cstdlib>


using namespace std;
using namespace openMVG;
using namespace openMVG::sfm;

int fullBA(std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir
)
{
    int iRotationAveragingMethod = int (ROTATION_AVERAGING_L2);
    int iTranslationAveragingMethod = int (TRANSLATION_AVERAGING_SOFTL1);
    std::string sIntrinsic_refinement_options = "ADJUST_ALL";
    std::string sMatchFilename;
    bool b_use_motion_priors = false;

    const cameras::Intrinsic_Parameter_Type intrinsic_refinement_options =
            cameras::StringTo_Intrinsic_Parameter_Type(sIntrinsic_refinement_options);

    // Load input SfM_Data scene
    SfM_Data sfm_data;
    if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS))) {
        std::cerr << std::endl
                  << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
        return 0;
    }

    // Init the regions_type from the image describer file (used for image regions extraction)
    using namespace openMVG::features;
    const std::string sImage_describer = stlplus::create_filespec(sMatchesDir, "image_describer", "json");
    std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
    if (!regions_type)
    {
        std::cerr << "Invalid: "
                  << sImage_describer << " regions type file." << std::endl;
        return 0;
    }

    // Features reading
    std::shared_ptr<Features_Provider> feats_provider = std::make_shared<Features_Provider>();
    if (!feats_provider->load(sfm_data, sMatchesDir, regions_type)) {
        std::cerr << std::endl
                  << "Invalid features." << std::endl;
        return 0;
    }
    // Matches reading
    std::shared_ptr<Matches_Provider> matches_provider = std::make_shared<Matches_Provider>();
    if // Try to read the provided match filename or the default one (matches.e.txt/bin)
            (
            !(matches_provider->load(sfm_data, sMatchFilename) ||
              matches_provider->load(sfm_data, stlplus::create_filespec(sMatchesDir, "matches.e.txt")) ||
              matches_provider->load(sfm_data, stlplus::create_filespec(sMatchesDir, "matches.e.bin")))
            )
    {
        std::cerr << std::endl
                  << "Invalid matches file." << std::endl;
        return 0;
    }

    if (!stlplus::folder_exists(sOutDir))
    {
        if (!stlplus::folder_create(sOutDir))
        {
            std::cerr << "\nCannot create the output directory" << std::endl;
        }
    }

    //---------------------------------------
    // Global SfM reconstruction process
    //---------------------------------------

    openMVG::system::Timer timer;
    GlobalSfMReconstructionEngine_RelativeMotions sfmEngine(
            sfm_data,
            sOutDir,
            stlplus::create_filespec(sOutDir, "Reconstruction_Report.html"));

    // Configure the features_provider & the matches_provider
    sfmEngine.SetFeaturesProvider(feats_provider.get());
    sfmEngine.SetMatchesProvider(matches_provider.get());

    // Configure reconstruction parameters
    sfmEngine.Set_Intrinsics_Refinement_Type(intrinsic_refinement_options);
    sfmEngine.Set_Use_Motion_Prior(b_use_motion_priors);

    // Configure motion averaging method
    sfmEngine.SetRotationAveragingMethod(
            ERotationAveragingMethod(iRotationAveragingMethod));
    sfmEngine.SetTranslationAveragingMethod(
            ETranslationAveragingMethod(iTranslationAveragingMethod));

    if (sfmEngine.Process())
    {
        std::cout << std::endl << " Total Ac-Global-Sfm took (s): " << timer.elapsed() << std::endl;
        std::cout << "...Generating SfM_Report.html" << std::endl;
        Generate_SfM_Report(sfmEngine.Get_SfM_Data(),
                            stlplus::create_filespec(sOutDir, "SfMReconstruction_Report.html"));
        Save(sfmEngine.Get_SfM_Data(),
             stlplus::create_filespec(sOutDir, "/ground_truth", ".bin"),
             ESfM_Data(ALL));
        return 1;
    }
    return 1;
}

void import_data(std::string sSfM_Data_Filename, double* poses3d, int d, double* intrinsics, int b,
                double* observation3d, int a, int* pcloud_idx, int h, double* observation2d, int f,
                int* camera, int c, int* track, int e, int* view, int vi, double* fullrot, int fr)
{
    SfM_Data sfm_data;
    if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(ALL))) {
        std::cerr << std::endl
                  << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
        return;
    }
    int pose_id = 0;
    for (auto & pose_it : sfm_data.poses)
    {
        const IndexT indexPose = pose_it.first;
        const Pose3 & pose = pose_it.second;
        const Vec3 t = pose.translation();
        const Mat3 R = pose.rotation();
        double angleAxis[3];
        sfm::extract_rotation(R, angleAxis);
        for (int jj = 0; jj < 3; jj++){poses3d[pose_id*6+jj+1] = angleAxis[jj];}
        for (int jj = 0; jj < 3; jj++){poses3d[pose_id*6+jj+4] = t(jj);}
        for (int jj = 0; jj < 9; jj++){fullrot[pose_id*9+jj+1] = R(jj);}
        view[pose_id] = indexPose;
        pose_id++;
    }

    poses3d[0] = sfm_data.poses.size();
    intrinsics[0] = sfm_data.intrinsics.size();
    for (const auto & intrinsic_it : sfm_data.intrinsics)
    {
        const IndexT indexCam = intrinsic_it.first;
        std::vector<double> param = intrinsic_it.second->getParams();
        int dim = param.size();
        for(int ii=0; ii<dim; ii++){intrinsics[indexCam * dim + ii + 1] = param[ii];}
    }

    int count = 0;
    int count2 = 0;
    for (auto & structure_landmark_it : sfm_data.structure)
    {
        const IndexT indexPose1 = structure_landmark_it.first;
        const Observations & obs = structure_landmark_it.second.obs;
        Eigen::Matrix<double,3,1> obs3d = structure_landmark_it.second.X;
        vector<double> obs_read3d(obs3d.data(), obs3d.data() + obs3d.size());
        pcloud_idx[count2+1] = indexPose1;
        for(int ii=0; ii<obs_read3d.size(); ii++){observation3d[count2 * obs_read3d.size() + ii] = obs_read3d[ii];}
        for (const auto & obs_it : obs)
        {
            const View * view = sfm_data.views.at(obs_it.first).get();
            const IndexT indexPose2 = obs_it.first;
            Eigen::Matrix<double,2,1> obs2d = obs_it.second.x;
            vector<double> obs_read(obs2d.data(), obs2d.data() + obs2d.size());
            for(int ii=0; ii<obs_read.size(); ii++){observation2d[count * 2 + ii] = obs_read[ii];}
            track [count+1] = count2;
            camera[count+1] = view->id_pose;
            count ++;
        }
        count2 ++;
    }

    track[0] = count;
    camera[0] = count;
    pcloud_idx[0] = count2;

    return;
}


void import_feats(std::string sSfM_Data_Filename, double* observation2d, int f, int* camera, int c, int* track,
                  int e, int* view, int vi)
{
    SfM_Data sfm_data;
    if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(ALL))) {
        std::cerr << std::endl
                  << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
        return;
    }
    int pose_id = 0;
    for (auto & pose_it : sfm_data.poses)
    {
        const IndexT indexPose = pose_it.first;
        const Pose3 & pose = pose_it.second;
        const Vec3 t = pose.translation();
        const Mat3 R = pose.rotation();
        double angleAxis[3];
        sfm::extract_rotation(R, angleAxis);
        view[pose_id] = indexPose;
        pose_id++;
    }

    int count = 0;
    int count2 = 0;
    for (auto & structure_landmark_it : sfm_data.structure)
    {
        const IndexT indexPose1 = structure_landmark_it.first;
        const Observations & obs = structure_landmark_it.second.obs;
        Eigen::Matrix<double,3,1> obs3d = structure_landmark_it.second.X;
        vector<double> obs_read3d(obs3d.data(), obs3d.data() + obs3d.size());
        for (const auto & obs_it : obs)
        {
            const View * view = sfm_data.views.at(obs_it.first).get();
            const IndexT indexPose2 = obs_it.first;
            Eigen::Matrix<double,2,1> obs2d = obs_it.second.x;
            vector<double> obs_read(obs2d.data(), obs2d.data() + obs2d.size());
            for(int ii=0; ii<obs_read.size(); ii++){observation2d[count * 2 + ii] = obs_read[ii];}
            track [count+1] = count2;
            camera[count+1] = view->id_pose;
            count ++;
        }
        count2 ++;
    }

    track[0] = count;
    camera[0] = count;
    return;
}


int CloudOnly(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
           int l_feat, int* cams, int l_cam, int* track, int l_track, double* residuals_out, int l_res)
{

    Bundle_Adjustment_Ceres bundle_adjustment_obj;

    int oo = bundle_adjustment_obj.CloudOnly(pose, l_pose, intrinsics, l_int, cloud, l_cloud, features, l_feat, cams,
                                          l_cam, track, l_track, residuals_out, l_res);
    return oo;
}


int BAonly(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
            int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
            int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
            int l_grad)
{

    Bundle_Adjustment_Ceres bundle_adjustment_obj;

    int oo = bundle_adjustment_obj.LastBA(pose, l_pose, intrinsics, l_int, cloud, l_cloud, features, l_feat, cams,
                                                l_cam, track, l_track, weight, l_w, vec_rows, l_vr, vec_cols, l_vc, vec_grad, l_vg, residuals_out, l_res,
                                                gradient_out, l_grad);
    return oo;
}

int BAonlyHuber(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
           int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
           int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
           int l_grad)
{

    Bundle_Adjustment_Ceres bundle_adjustment_obj;

    int oo = bundle_adjustment_obj.LastBAHuber(pose, l_pose, intrinsics, l_int, cloud, l_cloud, features, l_feat, cams,
                                          l_cam, track, l_track, weight, l_w, vec_rows, l_vr, vec_cols, l_vc, vec_grad, l_vg, residuals_out, l_res,
                                          gradient_out, l_grad);
    return oo;
}

int ceresBA(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
        int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
        int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
        int l_grad, int optimise)
{

    Bundle_Adjustment_Ceres bundle_adjustment_obj;

    int oo = bundle_adjustment_obj.GlobalAdjust(pose, l_pose, intrinsics, l_int, cloud, l_cloud, features, l_feat, cams,
            l_cam, track, l_track, weight, l_w, vec_rows, l_vr, vec_cols, l_vc, vec_grad, l_vg, residuals_out, l_res,
            gradient_out, l_grad, optimise);
    return oo;
}

int ceresBAHuber(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
            int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
            int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
            int l_grad, int optimise)
{

    Bundle_Adjustment_Ceres bundle_adjustment_obj;

    int oo = bundle_adjustment_obj.GlobalAdjustHuber(pose, l_pose, intrinsics, l_int, cloud, l_cloud, features, l_feat, cams,
                                                l_cam, track, l_track, weight, l_w, vec_rows, l_vr, vec_cols, l_vc, vec_grad, l_vg, residuals_out, l_res,
                                                gradient_out, l_grad, optimise);
    return oo;
}

int ceresBAAdaptive(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
                 int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
                 int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
                 int l_grad, int optimise, double shape, double scale)
{

    Bundle_Adjustment_Ceres bundle_adjustment_obj;

    int oo = bundle_adjustment_obj.GlobalAdjustAdaptive(pose, l_pose, intrinsics, l_int, cloud, l_cloud, features, l_feat, cams,
                                                     l_cam, track, l_track, weight, l_w, vec_rows, l_vr, vec_cols, l_vc, vec_grad, l_vg, residuals_out, l_res,
                                                     gradient_out, l_grad, optimise, shape, scale);
    return oo;
}