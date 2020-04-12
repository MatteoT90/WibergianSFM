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


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* poses3d, int po)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* intrinsics, int in)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* observation3d, int o1)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* observation2d, int o2)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* camera, int ca)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* track, int tr)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* pcloud_idx, int pc)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* vec_grad, int vg)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* vec_cols, int vc)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* vec_rows, int vr)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int k)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* cost, int m)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* residuals, int n)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* gradient, int o)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* rotations, int p)}

%include "std_string.i"
%include "std_vector.i"
%template(DoubleVector) std::vector<double>;
%template(DDVector) std::vector<std::vector<double>>;
%template(IntVector) std::vector<int>;

int fullBA( std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir);

void importdata(std::string sSfM_Data_Filename, double* poses3d, int po, double* intrinsics, int in,
        double* observation3d, int o1, int* pcloud_idx, int pc, double* observation2d, int o2, int* camera, int ca, int* track, int tr);
%newobject ceresBA;
int ceresBA(double* poses3d, int po, double* intrinsics, int in, double* observation3d,
            int o1, double* observation2d, int o2, int* camera, int ca, int* track, int tr,
            int* vec_rows, int vr, int* vec_cols, int vc, double* vec_grad, int vg, double* weights, int k,
            double* cost, int m, double* residuals, int n, double* gradient, int o, double* rotations, int p, int mode);

int preprocessing(std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir, std::string dataname);

int sanity_BA(std::string sSfM_Data_Filename, std::string sOutDir, std::string dataname, int* vec_rows, int vr, int* vec_cols, int vc, double* vec_grad, int vg);
