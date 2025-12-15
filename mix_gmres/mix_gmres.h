#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>

// Structure to hold CSR matrix on Device
struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    double* d_val;
    int* d_row_ptr;
    int* d_col_ind;
};

class MixedPrecisionGMRES {
public:
    MixedPrecisionGMRES(int n, int nnz, const CSRMatrix& A);
    ~MixedPrecisionGMRES();

    // Solve Ax = b
    // d_b: input vector (device, double)
    // d_x: initial guess / output vector (device, double)
    // restart: restart dimension (m)
    // max_iters: maximum number of outer iterations
    // tol: relative tolerance
    // method: orthogonalization method (0: MGS, 1: CGS)
    enum OrthogonalizationMethod { MGS = 0, CGS = 1 };
    void solve(const double* d_b, double* d_x, int restart, int max_iters, double tol, OrthogonalizationMethod method);

private:
    int n;
    int nnz;
    CSRMatrix A_high; // Reference to input matrix (double)
    
    // Low precision matrix data
    float* d_val_low;
    // We can reuse row_ptr and col_ind from A_high if we don't change structure, 
    // but cuSPARSE might need them to be consistent types? 
    // cuSPARSE uses int for indices, so we can reuse them.
    
    // Handles
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    // Descriptors
    cusparseSpMatDescr_t matA_high_descr;
    cusparseSpMatDescr_t matA_low_descr;
    cusparseDnVecDescr_t vec_x_high_descr;
    cusparseDnVecDescr_t vec_z_high_descr; // Residual
    cusparseDnVecDescr_t vec_b_high_descr;

    // Buffers for SpMV
    void* d_buffer_mv_high;
    size_t buffer_size_mv_high;
    void* d_buffer_mv_low;
    size_t buffer_size_mv_low;

    // Helper to convert A to float
    void convert_matrix_to_float();
};
