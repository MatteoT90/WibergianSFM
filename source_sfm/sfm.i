/* sfm.i */
%module sfm
%{
#define SWIG_FILE_WITH_INIT
#include "sfm.hpp"
%}

%include "numpy.i"

%init %{
    import_array();
%}


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* pose, int l_pose)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* intrinsics, int l_int)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* cloud, int l_cloud)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* features, int l_feat)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* camera, int l_cam)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* track, int l_track)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* pcloud_idx, int pc)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* vec_grad, int l_vg)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* vec_cols, int l_vc)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* vec_rows, int l_vr)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int l_w)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* cost, int l_cost)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* residuals, int l_res)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* gradient, int l_grad)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* rotations, int p)}

%include "std_string.i"
%include "std_vector.i"
%template(DoubleVector) std::vector<double>;
%template(DDVector) std::vector<std::vector<double>>;
%template(IntVector) std::vector<int>;

int fullBA( std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir);

void import_data(std::string sSfM_Data_Filename, double* poses3d, int po, double* intrinsics, int in,
        double* observation3d, int o1, int* pcloud_idx, int pc, double* observation2d, int o2, int* camera, int ca,
        int* track, int tr);
%newobject ceresBA;
int ceresBA(double* pose, int l_pose, double* intrinsics, int l_int, double* cloud, int l_cloud, double* features,
        int l_feat, int* cams, int l_cam, int* track, int l_track, double* weight, int l_w, int* vec_rows, int l_vr,
        int* vec_cols, int l_vc, double* vec_grad, int l_vg, double* residuals_out, int l_res, double* gradient_out,
        int l_grad, int optimise);

int sanity_BA(std::string sSfM_Data_Filename, std::string sOutDir, std::string data_name, int* vec_rows,
        int vr, int* vec_cols, int vc, double* vec_grad, int vg);
