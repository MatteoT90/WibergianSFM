#include <string>
#include <vector>
int fullBA(std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir);

void importdata(std::string sSfM_Data_Filename, double* poses3d, int d, double* intrinsics, int b,
        double* observation3d, int a, int* pcloud_idx, int h, double* observation2d, int f, int* camera, int c, int* track, int e);

int ceresBA(double* poses3d, int a, double* intrinsics, int b, double* observation3d,
            int c, double* observation2d, int e, int* camera, int f, int* track, int g,
            int* vec_rows, int h, int* vec_cols, int i, double* vec_grad, int j, double* weights, int k,
            double* cost, int m, double* residuals, int n, double* gradient, int o, double* rotation, int p, int mode);

int preprocessing(std::string sSfM_Data_Filename, std::string sMatchesDir, std::string sOutDir, std::string dataname);

int sanity_BA(std::string sSfM_Data_Filename, std::string sOutDir, std::string dataname, int* vec_rows, int h, int* vec_cols, int i, double* vec_grad, int j);