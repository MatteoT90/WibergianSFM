#include <string>
#include <vector>
int fullBA(std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir);

void import_data(std::string sSfM_Data_Filename, double* poses3d, int d, double* intrinsics, int b,
        double* observation3d, int a, int* pcloud_idx, int h, double* observation2d, int f, int* camera, int c,
        int* track, int e, int* view, int vi, double* fullrot, int fr);

void import_feats(std::string sSfM_Data_Filename, double* observation2d, int f, int* camera, int c, int* track,
        int e, int* view, int vi);

int CloudOnly(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
              int l_feat, int* cams, int l_cam, int* track, int l_track, double* residuals_out, int l_res);

int ceresBA(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
        int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
        int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
        int l_grad, int optimise);

int BAonly(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
            int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
            int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
            int l_grad);

int BAonlyHuber(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
           int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
           int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
           int l_grad);

int ceresBAHuber(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
            int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
            int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
            int l_grad, int optimise);

int ceresBAAdaptive(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
                 int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
                 int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
                 int l_grad, int optimise, double shape, double scale);