#include "mix_gmres.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>

// Helper to read MTX file
bool readMTXFile(const char* filename, int& rows, int& cols, int& nnz, 
                 std::vector<int>& row_ptr, std::vector<int>& col_ind, std::vector<double>& val) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Cannot open file %s\n", filename);
        return false;
    }
    
    std::string line;
    // Skip header and comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');
    
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;
    
    std::vector<int> row_indices(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<double> values(nnz);
    
    for (int i = 0; i < nnz; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> row_indices[i] >> col_indices[i] >> values[i];
        row_indices[i]--; // Convert to 0-based indexing
        col_indices[i]--;
    }
    
    file.close();
    
    // Convert to CSR format
    row_ptr.resize(rows + 1, 0);
    col_ind.resize(nnz);
    val.resize(nnz);
    
    // Count entries in each row
    for (int i = 0; i < nnz; i++) {
        row_ptr[row_indices[i] + 1]++;
    }
    
    // Convert counts to offsets
    for (int i = 1; i <= rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }
    
    // Fill CSR arrays
    std::vector<int> temp_row_ptr = row_ptr;
    for (int i = 0; i < nnz; i++) {
        int row = row_indices[i];
        int dest = temp_row_ptr[row]++;
        col_ind[dest] = col_indices[i];
        val[dest] = values[i];
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_file.mtx> [orth_method] [restart] [max_iters] [tol]\n", argv[0]);
        printf("orth_method: mgs (default) or cgs\n");
        printf("Defaults: restart=30, max_iters=100, tol=1e-6\n");
        return 1;
    }

    MixedPrecisionGMRES::OrthogonalizationMethod method = MixedPrecisionGMRES::MGS;
    if (argc >= 3) {
        std::string method_str = argv[2];
        if (method_str == "cgs" || method_str == "CGS") {
            method = MixedPrecisionGMRES::CGS;
            printf("Using CGS orthogonalization\n");
        } else {
            printf("Using MGS orthogonalization (default)\n");
        }
    } else {
        printf("Using MGS orthogonalization (default)\n");
    }

    int restart = 30;
    if (argc >= 4) restart = atoi(argv[3]);

    int max_iters = 100;
    if (argc >= 5) max_iters = atoi(argv[4]);

    double tol = 1e-6;
    if (argc >= 6) tol = atof(argv[5]);

    printf("Parameters: restart=%d, max_iters=%d, tol=%e\n", restart, max_iters, tol);

    int n, cols, nnz;
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_ind;
    std::vector<double> h_val;

    printf("Reading matrix %s...\n", argv[1]);
    if (!readMTXFile(argv[1], n, cols, nnz, h_row_ptr, h_col_ind, h_val)) {
        return 1;
    }

    if (n != cols) {
        printf("Matrix must be square!\n");
        return 1;
    }

    printf("Matrix loaded: %d x %d, %d nnz\n", n, cols, nnz);

    // Allocate Device Memory for Matrix
    CSRMatrix A_dev;
    A_dev.rows = n;
    A_dev.cols = n;
    A_dev.nnz = nnz;

    cudaMalloc(&A_dev.d_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&A_dev.d_col_ind, (size_t)nnz * sizeof(int));
    cudaMalloc(&A_dev.d_val, (size_t)nnz * sizeof(double));

    cudaMemcpy(A_dev.d_row_ptr, h_row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_dev.d_col_ind, h_col_ind.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_dev.d_val, h_val.data(), (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice);

    // Setup RHS b and Initial Guess x
    // Let's set true solution x_true = 1, and b = A * x_true
    // Then initial guess x = 0
    
    double* d_x_true;
    cudaMalloc(&d_x_true, (size_t)n * sizeof(double));
    std::vector<double> h_x_true(n, 1.0);
    cudaMemcpy(d_x_true, h_x_true.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice);

    double* d_b;
    cudaMalloc(&d_b, (size_t)n * sizeof(double));
    
    // Compute b = A * x_true using cuSPARSE (or just do it simply here)
    // We can use the solver class to do SpMV if we exposed it, but let's just use a temporary handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, n, n, nnz, A_dev.d_row_ptr, A_dev.d_col_ind, A_dev.d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseDnVecDescr_t vecX, vecB;
    cusparseCreateDnVec(&vecX, n, d_x_true, CUDA_R_64F);
    cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_64F);
    
    double alpha = 1.0, beta = 0.0;
    size_t bufferSize;
    void* d_buffer;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecB, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&d_buffer, bufferSize);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecB, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    
    cudaFree(d_buffer);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecB);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);

    // Initial guess x = 0
    double* d_x;
    cudaMalloc(&d_x, (size_t)n * sizeof(double));
    cudaMemset(d_x, 0, (size_t)n * sizeof(double));

    // Solve
    printf("Starting Mixed Precision GMRES...\n");
    MixedPrecisionGMRES solver(n, nnz, A_dev);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    solver.solve(d_b, d_x, restart, max_iters, tol, method);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Solver Time: %.6f seconds\n", elapsed.count());

    // Compute final relative residual ||b - A*x_final|| / ||b - A*x_0||
    // Since x_0 = 0, ||b - A*x_0|| = ||b||
    
    // Re-create handles for residual check
    cusparseHandle_t handle_check;
    cusparseCreate(&handle_check);
    cublasHandle_t cublas_check;
    cublasCreate(&cublas_check);
    
    cusparseSpMatDescr_t matA_check;
    cusparseCreateCsr(&matA_check, n, n, nnz, A_dev.d_row_ptr, A_dev.d_col_ind, A_dev.d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    double* d_r_final;
    cudaMalloc(&d_r_final, (size_t)n * sizeof(double));
    // r_final = b
    cudaMemcpy(d_r_final, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice);
    
    cusparseDnVecDescr_t vecX_final, vecR_final;
    cusparseCreateDnVec(&vecX_final, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecR_final, n, d_r_final, CUDA_R_64F);
    
    // r_final = b - A * x_final
    double alpha_check = -1.0;
    double beta_check = 1.0;
    size_t bufferSize_check;
    void* d_buffer_check;
    
    cusparseSpMV_bufferSize(handle_check, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                           &alpha_check, matA_check, vecX_final, &beta_check, vecR_final, 
                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_check);
    cudaMalloc(&d_buffer_check, bufferSize_check);
    
    cusparseSpMV(handle_check, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &alpha_check, matA_check, vecX_final, &beta_check, vecR_final, 
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_check);
                 
    double norm_r_final;
    cublasDnrm2(cublas_check, n, d_r_final, 1, &norm_r_final);
    
    double norm_b; // This is ||r_0|| since x_0 = 0
    cublasDnrm2(cublas_check, n, d_b, 1, &norm_b);
    
    printf("Final Relative Residual ||b - Ax|| / ||b||: %e\n", norm_r_final / norm_b);
    
    cudaFree(d_buffer_check);
    cudaFree(d_r_final);
    cusparseDestroyDnVec(vecX_final);
    cusparseDestroyDnVec(vecR_final);
    cusparseDestroySpMat(matA_check);
    cusparseDestroy(handle_check);
    cublasDestroy(cublas_check);

    // Check error
    // We can compute norm(x - x_true)
    double* h_x_result = new double[n];
    cudaMemcpy(h_x_result, d_x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double error_norm = 0.0;
    for(int i=0; i<n; ++i) {
        double diff = h_x_result[i] - h_x_true[i];
        error_norm += diff * diff;
    }
    error_norm = sqrt(error_norm);
    printf("Final Error Norm ||x - x_true||: %e\n", error_norm);

    // Cleanup
    delete[] h_x_result;
    cudaFree(d_x_true);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(A_dev.d_row_ptr);
    cudaFree(A_dev.d_col_ind);
    cudaFree(A_dev.d_val);

    return 0;
}
