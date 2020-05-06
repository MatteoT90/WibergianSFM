#include <string>
#include <vector>
int fullBA(std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir);

void import_data(std::string sSfM_Data_Filename, double* poses3d, int d, double* intrinsics, int b,
        double* observation3d, int a, int* pcloud_idx, int h, double* observation2d, int f, int* camera, int c,
        int* track, int e);

int ceresBA(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud,
    int l_cloud, double* features, int l_feat, int* camera, int l_cam, int* track, int l_track, double* weights, int l_w,
    int* vec_rows, int l_vr, int* vec_cols, int l_vc, double* vec_grad, int l_vg,
    double* cost, int l_cost, double* residuals, int l_res, double* gradient, int l_grad, int run)

int sanity_BA(std::string sSfM_Data_Filename, std::string sOutDir, std::string data_name, int* vec_rows, int h,
        int* vec_cols, int i, double* vec_grad, int j);
