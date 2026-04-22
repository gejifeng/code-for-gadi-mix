#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <vector>

struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    double* d_val;
    int* d_row_ptr;
    int* d_col_ind;
};

class FP64GMRES {
public:
    FP64GMRES(int n, int nnz, const CSRMatrix& A);
    ~FP64GMRES();

    enum OrthogonalizationMethod { MGS = 0, CGS = 1 };

    void solve(const double* d_b, double* d_x, int restart, int max_iters, double tol,
               OrthogonalizationMethod method);

private:
    int n;
    int nnz;
    CSRMatrix A;

    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cusparseSpMatDescr_t matA_descr;

    void* d_buffer_mv;
    size_t buffer_size_mv;
};