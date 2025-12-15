#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s %d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE error at %s %d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// Structure to hold sparse matrix in CSR format
struct SparseMatrix {
    int rows, cols, nnz;
    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal;

    SparseMatrix() : rows(0), cols(0), nnz(0), d_csrRowPtr(nullptr), d_csrColInd(nullptr), d_csrVal(nullptr) {}

    ~SparseMatrix() {
        cleanup();
    }

    void cleanup() {
        if (d_csrRowPtr) cudaFree(d_csrRowPtr);
        if (d_csrColInd) cudaFree(d_csrColInd);
        if (d_csrVal) cudaFree(d_csrVal);
        d_csrRowPtr = nullptr;
        d_csrColInd = nullptr;
        d_csrVal = nullptr;
    }
};


// Function to read MTX file
bool readMTXFile(const char* filename, SparseMatrix &matrix) {
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
    iss >> matrix.rows >> matrix.cols >> matrix.nnz;
    
    std::vector<int> row_indices(matrix.nnz);
    std::vector<int> col_indices(matrix.nnz);
    std::vector<double> values(matrix.nnz);
    
    for (int i = 0; i < matrix.nnz; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> row_indices[i] >> col_indices[i] >> values[i];
        row_indices[i]--; // Convert to 0-based indexing
        col_indices[i]--;
    }
    
    file.close();
    
    // Convert to CSR format
    std::vector<int> csr_row_ptr(matrix.rows + 1, 0);
    std::vector<int> csr_col_ind(matrix.nnz);
    std::vector<double> csr_val(matrix.nnz);
    
    // Count entries in each row
    for (int i = 0; i < matrix.nnz; i++) {
        csr_row_ptr[row_indices[i] + 1]++;
    }
    
    // Convert counts to offsets
    for (int i = 1; i <= matrix.rows; i++) {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }
    
    std::vector<int> temp_row_ptr = csr_row_ptr;
    
    // Fill CSR arrays
    for (int i = 0; i < matrix.nnz; i++) {
        int row = row_indices[i];
        if (row < 0 || row >= matrix.rows) {
            printf("Error: Invalid row index %d (should be 0-%d)\n", row, matrix.rows-1);
            return false;
        }
        int pos = temp_row_ptr[row]++;
        if (pos < 0 || pos >= matrix.nnz) {
            printf("Error: Invalid position %d (should be 0-%d)\n", pos, matrix.nnz-1);
            return false;
        }
        csr_col_ind[pos] = col_indices[i];
        csr_val[pos] = values[i];
    }
    
    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(&matrix.d_csrRowPtr, (matrix.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&matrix.d_csrColInd, matrix.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&matrix.d_csrVal, matrix.nnz * sizeof(double)));
    
    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(matrix.d_csrRowPtr, csr_row_ptr.data(), (matrix.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_csrColInd, csr_col_ind.data(), matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_csrVal, csr_val.data(), matrix.nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    return true;
}

// CG solver in double precision
int cg_double_precision(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                        SparseMatrix &A, double *d_b, double *d_x,
                        double tol = 1e-4, int max_iter = 100) {
    
    int n = A.rows;
    double *d_r, *d_p, *d_Ap;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(double)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    double rsold, rsnew, alpha_cg, beta;
    double one = 1.0, zero = 0.0;
    
    // rsold = r^T * r
    CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsold));
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F));

    cusparseSpMatDescr_t A_mat_double;
    CHECK_CUSPARSE(cusparseCreateCsr(&A_mat_double, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    int iter;

    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, A_mat_double, vecP, &zero, vecAp,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, A_mat_double, vecP, &zero, vecAp,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        // alpha = rsold / (p^T * Ap)
        double pAp;
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_p, 1, d_Ap, 1, &pAp));
        alpha_cg = rsold / pAp;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha_cg, d_p, 1, d_x, 1));
        
        // r = r - alpha * Ap
        double neg_alpha = -alpha_cg;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ap, 1, d_r, 1));
        
        // rsnew = r^T * r
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsnew));
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        // beta = rsnew / rsold
        beta = rsnew / rsold;
        
        // p = r + beta * p
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &beta, d_p, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_p, 1));
        
        rsold = rsnew;
    }
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_mat_double));

//    printf("CG half precision completed in %d iterations with residual %e\n", iter, sqrt(rsnew));
    
    return iter;
}

// CG solver in double precision
int cg_double_precision1(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                        SparseMatrix &AIMS, SparseMatrix &AIPS, double *d_b, double *d_x,
                        double tol = 1e-4, int max_iter = 100) {
    
    int n = AIMS.rows;
    double *d_r, *d_p, *d_Ap, *d_tmp;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(double)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    double rsold, rsnew, alpha_cg, beta;
    double one = 1.0, zero = 0.0;
    
    // rsold = r^T * r
    CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsold));
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp, vecTmp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTmp, n, d_tmp, CUDA_R_64F));
    cusparseSpMatDescr_t AIPS_mat, AIMS_mat;
    CHECK_CUSPARSE(cusparseCreateCsr(&AIPS_mat, AIPS.rows, AIPS.cols, AIPS.nnz,
                                     AIPS.d_csrRowPtr, AIPS.d_csrColInd, AIPS.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&AIMS_mat, AIMS.rows, AIMS.cols, AIMS.nnz,
                                     AIMS.d_csrRowPtr, AIMS.d_csrColInd, AIMS.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    int iter;

    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    // Get buffer size
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, AIPS_mat, vecP, &zero, vecTmp,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, AIMS_mat, vecTmp, &zero, vecAp,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, AIPS_mat, vecP, &zero, vecTmp,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, AIMS_mat, vecTmp, &zero, vecAp,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));
        
        // alpha = rsold / (p^T * Ap)
        double pAp;
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_p, 1, d_Ap, 1, &pAp));
        alpha_cg = rsold / pAp;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha_cg, d_p, 1, d_x, 1));
        
        // r = r - alpha * Ap
        double neg_alpha = -alpha_cg;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ap, 1, d_r, 1));
        
        // rsnew = r^T * r
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsnew));
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        // beta = rsnew / rsold
        beta = rsnew / rsold;
        
        // p = r + beta * p
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &beta, d_p, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_p, 1));
        
        rsold = rsnew;
    }
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_tmp);
    if (dBuffer) cudaFree(dBuffer);
    if (dBuffer1) cudaFree(dBuffer1);
    CHECK_CUSPARSE(cusparseDestroySpMat(AIPS_mat));
    CHECK_CUSPARSE(cusparseDestroySpMat(AIMS_mat));

//    printf("CG half precision completed in %d iterations with residual %e\n", iter, sqrt(rsnew));
    
    return iter;
}

// Double-precision iterative solver
int double_precision_solver(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                           SparseMatrix &A, SparseMatrix &AH, SparseMatrix &AIMS, SparseMatrix &AIPS, SparseMatrix &S,
                           double *d_b, double *d_x0, double w, double alpha_param,
                           double tol = 1e-6, int max_iter = 3000) {
    
    int n = A.rows;
    double *d_r, *d_r1, *d_r2, *d_r3;
    double *d_temp, *d_eye_r1;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r1, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r2, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r3, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_temp, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_eye_r1, n * sizeof(double)));
    
    double nr0;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_b, 1, &nr0));
    
    double res = 1.0;
    int kk = 0;
    long long total_cg_iters = 0;

    cusparseDnVecDescr_t vecX0, vecR;
    cusparseDnVecDescr_t vecR1, vecTemp;
    
    // Create vector descriptors once before the loop
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX0, n, d_x0, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR1, n, d_r1, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTemp, n, d_temp, CUDA_R_64F));

    cusparseSpMatDescr_t matA, matS;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matS, S.rows, S.cols, S.nnz,
                                     S.d_csrRowPtr, S.d_csrColInd, S.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    double one = 1.0, zero = 0.0, neg_one = -1.0;

    // spmv buffer size for A
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &neg_one, matA, vecX0, &one, vecR,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // spmv buffer size for S
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &one, matS, vecR1, &zero, vecTemp,
                                               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    while (res > tol && kk < max_iter) {

        CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &neg_one, matA, vecX0, &one, vecR,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    
        // r1 = CG_solve(AH, r) in double precision
        total_cg_iters += cg_double_precision(cusparseHandle, cublasHandle, AH, d_r, d_r1);
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        // First: eye_r1 = alpha * I * r1 = alpha * r1
        CHECK_CUDA(cudaMemcpy(d_eye_r1, d_r1, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &alpha_param, d_eye_r1, 1));
        
        // Then: temp = S * r1
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, matS, vecR1, &zero, vecTemp,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));
        
        // r2 = alpha * r1 - S * r1 = eye_r1 - temp
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &neg_one, d_temp, 1, d_eye_r1, 1));
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        double factor = (2.0 - w) * alpha_param;
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &factor, d_eye_r1, 1));
        CHECK_CUDA(cudaMemcpy(d_r2, d_eye_r1, n * sizeof(double), cudaMemcpyDeviceToDevice));
        
        // r3 = CG_solve(ASS, r2) in double precision
        total_cg_iters += cg_double_precision1(cusparseHandle, cublasHandle, AIMS, AIPS, d_r2, d_r3);
        
        // x0 = x0 + r3
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r3, 1, d_x0, 1));
        
        // Check convergence
        double r_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_r, 1, &r_norm));
        res = r_norm / nr0;
        
        kk++;
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX0));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR1));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecTemp));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matS));
    if (dBuffer) cudaFree(dBuffer);
    if (dBuffer1) cudaFree(dBuffer1);
    
    printf("Double-precision solver: res = %e, iter = %d\n", res, kk);
    printf("Total CG iterations: %lld\n", total_cg_iters);
    
    // Cleanup
    cudaFree(d_r);
    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_r3);
    cudaFree(d_temp);
    cudaFree(d_eye_r1);
    
    return kk;
}

// Function to create identity matrix in CSR format
void createIdentityMatrix(SparseMatrix &I, int n) {
    I.rows = n;
    I.cols = n;
    I.nnz = n;
    
    CHECK_CUDA(cudaMalloc(&I.d_csrRowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&I.d_csrColInd, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&I.d_csrVal, n * sizeof(double)));
    
    std::vector<int> row_ptr(n + 1);
    std::vector<int> col_ind(n);
    std::vector<double> val(n, 1.0);
    
    for (int i = 0; i <= n; i++) {
        row_ptr[i] = i;
    }
    for (int i = 0; i < n; i++) {
        col_ind[i] = i;
    }
    
    CHECK_CUDA(cudaMemcpy(I.d_csrRowPtr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(I.d_csrColInd, col_ind.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(I.d_csrVal, val.data(), n * sizeof(double), cudaMemcpyHostToDevice));
}

// Function to transpose a sparse matrix (proper implementation)
void transposeMatrix(cusparseHandle_t cusparseHandle, const SparseMatrix &A, SparseMatrix &AT) {
    AT.rows = A.cols;
    AT.cols = A.rows;
    AT.nnz = A.nnz;
    
    // Allocate memory for AT
    CHECK_CUDA(cudaMalloc(&AT.d_csrRowPtr, (AT.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&AT.d_csrColInd, AT.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&AT.d_csrVal, AT.nnz * sizeof(double)));
    
    // Use cuSPARSE transpose operation
    // First, create temporary buffers for the transpose operation
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = nullptr;
    
    // Get buffer size for transpose operation
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(cusparseHandle, A.rows, A.cols, A.nnz,
                                                 A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                                 AT.d_csrVal, AT.d_csrRowPtr, AT.d_csrColInd,
                                                 CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG1, &pBufferSizeInBytes));
    
    // Allocate buffer
    if (pBufferSizeInBytes > 0) {
        CHECK_CUDA(cudaMalloc(&pBuffer, pBufferSizeInBytes));
    }
    
    // Perform the transpose operation (CSR to CSC, which is equivalent to transpose)
    CHECK_CUSPARSE(cusparseCsr2cscEx2(cusparseHandle, A.rows, A.cols, A.nnz,
                                      A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                      AT.d_csrVal, AT.d_csrRowPtr, AT.d_csrColInd,
                                      CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, pBuffer));
    
    // Clean up buffer
    if (pBuffer) {
        cudaFree(pBuffer);
        
    }
    printf("Matrix transpose completed: %d x %d -> %d x %d\n", A.rows, A.cols, AT.rows, AT.cols);
}

// 类型特征：检查是否需要half精度转换
void matrixAdd(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
               const SparseMatrix &A, const SparseMatrix &B, SparseMatrix &C,
               double scaleA = 1.0, double scaleB = 1.0) {
    
    // Set pointer mode to host
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
    
    // Initialize result matrix dimensions
    C.rows = A.rows;
    C.cols = A.cols;
    
    // Step 1: Allocate workspace buffer
    size_t bufferSizeInBytes = 0;
    void *pBuffer = nullptr;
    
    // Create matrix descriptors for legacy API
    cusparseMatDescr_t descrA, descrB, descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    
    // Allocate row pointer for result matrix
    CHECK_CUDA(cudaMalloc(&C.d_csrRowPtr, (C.rows + 1) * sizeof(int)));
    
    // Get buffer size for csrgeam2
    CHECK_CUSPARSE(cusparseDcsrgeam2_bufferSizeExt(cusparseHandle, C.rows, C.cols,
                                                   &scaleA, descrA, A.nnz, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                                   &scaleB, descrB, B.nnz, B.d_csrVal, B.d_csrRowPtr, B.d_csrColInd,
                                                   descrC, nullptr, C.d_csrRowPtr, nullptr,
                                                   &bufferSizeInBytes));
    
    // Allocate workspace buffer
    if (bufferSizeInBytes > 0) {
        CHECK_CUDA(cudaMalloc(&pBuffer, bufferSizeInBytes));
    }
    
    // Step 2: Determine number of non-zeros in result
    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(cusparseHandle, C.rows, C.cols,
                                        descrA, A.nnz, A.d_csrRowPtr, A.d_csrColInd,
                                        descrB, B.nnz, B.d_csrRowPtr, B.d_csrColInd,
                                        descrC, C.d_csrRowPtr, nnzTotalDevHostPtr, pBuffer));
    
    C.nnz = nnzC;
    
    // Step 3: Allocate memory for column indices and values
    CHECK_CUDA(cudaMalloc(&C.d_csrColInd, C.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&C.d_csrVal, C.nnz * sizeof(double)));
    
    // Step 4: Perform the actual addition: C = scaleA * A + scaleB * B
    CHECK_CUSPARSE(cusparseDcsrgeam2(cusparseHandle, C.rows, C.cols,
                                     &scaleA, descrA, A.nnz, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                     &scaleB, descrB, B.nnz, B.d_csrVal, B.d_csrRowPtr, B.d_csrColInd,
                                     descrC, C.d_csrVal, C.d_csrRowPtr, C.d_csrColInd, pBuffer));
    
    // Cleanup
    if (pBuffer) cudaFree(pBuffer);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrB);
    cusparseDestroyMatDescr(descrC);
}

// Function to print memory usage
void printMemoryUsage() {
    size_t free_bytes, total_bytes;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    size_t used_bytes = total_bytes - free_bytes;
    printf("GPU Memory Usage:\n");
    printf("  Used: %.2f MB\n", used_bytes / 1024.0 / 1024.0);
    printf("  Free: %.2f MB\n", free_bytes / 1024.0 / 1024.0);
    printf("  Total: %.2f MB\n", total_bytes / 1024.0 / 1024.0);
}

int main(int argc, char **argv) {

    if (argc != 4) {
        printf("Usage: %s <mtx_file> <alpha> <w>\n", argv[0]);
        printf("  mtx_file: path to matrix market file\n");
        printf("  alpha: alpha parameter (e.g., 0.48)\n");
        printf("  w: w parameter (e.g., 0.1)\n");
        return 1;
    }

    // Initialize CUDA libraries
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    
    // Parse command line arguments
    const char* mtx_file = argv[1];
    double alpha = atof(argv[2]);
    double w = atof(argv[3]);
    
    // Validate parameters
    if (alpha <= 0 || alpha >= 1) {
        printf("Warning: alpha = %f is outside typical range (0, 1)\n", alpha);
    }
    if (w <= 0 || w >= 2) {
        printf("Warning: w = %f is outside typical range (0, 2)\n", w);
    }
    
    printf("Parameters: alpha = %f, w = %f\n", alpha, w);
    
    // Read matrix from MTX file
    SparseMatrix A;
    if (!readMTXFile(mtx_file, A)) {
        printf("Failed to read MTX file\n");
        return 1;
    }
    
    printf("Matrix dimensions: %d x %d, nnz = %d\n", A.rows, A.cols, A.nnz);
    
    // Parameters (matching the Python code)
    // double alpha = 0.9;
    // double w = 0.6;
    int n = A.rows;
    
    // Create identity matrix
    SparseMatrix I;
    createIdentityMatrix(I, n);
    
    // Compute the required matrices according to the algorithm:
    // h = 0.5 * (A + A^T)  // Symmetric part (double)
    // s = 0.5 * (A - A^T)  // Skew-symmetric part (double)
    // ah = alpha * I + h   // AH matrix (two precision)
    // asx = alpha * I + s  // Auxiliary matrix (double)
    // ass = (alpha * I - s) @ asx  // ASS matrix (two precision)
    // (alpha * I - s) // (two precision)

    printf("Computing required matrices...\n");
    
    // Step 1: Compute A^T (transpose of A)
    SparseMatrix AT;
    transposeMatrix(cusparseHandle, A, AT);
    
    // Step 2: Compute h = 0.5 * (A + A^T) - symmetric part
    SparseMatrix H;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, H, 0.5, 0.5);
    
    // Step 3: Compute s = 0.5 * (A - A^T) - skew-symmetric part  
    SparseMatrix S;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, S, 0.5, -0.5);
    
    // Step 4: Compute AH = alpha * I + h
    SparseMatrix AH;
    matrixAdd(cusparseHandle, cublasHandle, I, H, AH, alpha, 1.0);
    
    // Step 5: Compute asx = alpha * I + s
    SparseMatrix ASX;
    matrixAdd(cusparseHandle, cublasHandle, I, S, ASX, alpha, 1.0);
    
    // Step 6: Compute alpha * I - s
    SparseMatrix ALPHA_I_MINUS_S;
    matrixAdd(cusparseHandle, cublasHandle, I, S, ALPHA_I_MINUS_S, alpha, -1.0);
    
    // // Step 7: Compute ASS = (alpha * I - s) @ asx
    // SparseMatrix ASS;
    // matrixMultiply(cusparseHandle, ALPHA_I_MINUS_S, ASX, ASS);
    
    printf("Matrix computation completed.\n");
    printf("H: %d x %d, nnz = %d\n", H.rows, H.cols, H.nnz);
    printf("S: %d x %d, nnz = %d\n", S.rows, S.cols, S.nnz);  
    printf("AH: %d x %d, nnz = %d\n", AH.rows, AH.cols, AH.nnz);
    // printf("ASS: %d x %d, nnz = %d\n", ASS.rows, ASS.cols, ASS.nnz);

    AT.cleanup();   // Manually release transpose matrix
    H.cleanup();    // Manually release symmetric part
    I.cleanup();  // Manually release identity matrix
    
    // Create right-hand side vector (b = A * ones)
    double *d_b, *d_x0;
    double *d_ones;
    
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x0, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_ones, n * sizeof(double)));
    
    // Initialize ones vector
    std::vector<double> ones(n, 1.0);
    CHECK_CUDA(cudaMemcpy(d_ones, ones.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    
    // b = A @ ones
    cusparseDnVecDescr_t vecOnes, vecB;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecOnes, n, d_ones, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_64F));
    
    double one = 1.0, zero = 0.0;
    size_t bufferSize;
    void *dBuffer = nullptr;

    cusparseSpMatDescr_t A_matDescr;
    CHECK_CUSPARSE(cusparseCreateCsr(&A_matDescr, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, A_matDescr, vecOnes, &zero, vecB,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }
    
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &one, A_matDescr, vecOnes, &zero, vecB,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_matDescr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecOnes));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));

    double res = 1e-6;
    
    // Initialize solution vectors to zero
    CHECK_CUDA(cudaMemset(d_x0, 0, n * sizeof(double)));
    CHECK_CUDA(cudaFree(d_ones));
    
    printMemoryUsage();
    
    // Run double-precision solver
    printf("\n=== Running Double-Precision Solver ===\n");
    
    cudaEvent_t start_double, stop_double;
    CHECK_CUDA(cudaEventCreate(&start_double));
    CHECK_CUDA(cudaEventCreate(&stop_double));
    
    CHECK_CUDA(cudaEventRecord(start_double));
    int iter_double = double_precision_solver(cusparseHandle, cublasHandle, A, AH, ALPHA_I_MINUS_S, ASX, S,
                                             d_b, d_x0, w, alpha, res, 3000);
    // int iter_double = 0;
    CHECK_CUDA(cudaEventRecord(stop_double));
    CHECK_CUDA(cudaEventSynchronize(stop_double));
    
    float time_double;
    CHECK_CUDA(cudaEventElapsedTime(&time_double, start_double, stop_double));
    
    printf("Double-precision iterations: %d\n", iter_double);
    printf("Double-precision time: %.4f seconds\n", time_double / 1000.0f);
    
    printMemoryUsage();
    
    // Cleanup
    cudaFree(d_b);
    cudaFree(d_x0);
    
    // Cleanup original matrices
    
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    
    CHECK_CUDA(cudaEventDestroy(start_double));
    CHECK_CUDA(cudaEventDestroy(stop_double));
    
    return 0;
}