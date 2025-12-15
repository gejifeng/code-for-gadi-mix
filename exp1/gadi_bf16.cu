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
struct SparseMatrixMixed {
    int rows, cols, nnz;
    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal;
    float *d_csrVal_bf16;
    
    // Constructor
    SparseMatrixMixed() : rows(0), cols(0), nnz(0), 
                         d_csrRowPtr(nullptr), d_csrColInd(nullptr), 
                         d_csrVal(nullptr), d_csrVal_bf16(nullptr) {}
    
    // Destructor
    ~SparseMatrixMixed() {
        cleanup();
    }
    
    // Manual cleanup method - can be called explicitly if needed
    void cleanup() {
        if (d_csrRowPtr) {
            cudaFree(d_csrRowPtr);
            d_csrRowPtr = nullptr;
        }
        if (d_csrColInd) {
            cudaFree(d_csrColInd);
            d_csrColInd = nullptr;
        }
        if (d_csrVal) {
            cudaFree(d_csrVal);
            d_csrVal = nullptr;
        }
        if (d_csrVal_bf16) {
            cudaFree(d_csrVal_bf16);
            d_csrVal_bf16 = nullptr;
        }
        rows = cols = nnz = 0;
    }
    
    // Check if matrix has been allocated
    bool isEmpty() const {
        return (d_csrRowPtr == nullptr && d_csrColInd == nullptr && 
                d_csrVal == nullptr && d_csrVal_bf16 == nullptr);
    }
    
    // Disable copy constructor and assignment operator to prevent double-free
    SparseMatrixMixed(const SparseMatrixMixed&) = delete;
    SparseMatrixMixed& operator=(const SparseMatrixMixed&) = delete;
    
    // Enable move constructor and assignment operator
    SparseMatrixMixed(SparseMatrixMixed&& other) noexcept 
        : rows(other.rows), cols(other.cols), nnz(other.nnz),
          d_csrRowPtr(other.d_csrRowPtr), d_csrColInd(other.d_csrColInd),
          d_csrVal(other.d_csrVal), d_csrVal_bf16(other.d_csrVal_bf16) {
        other.d_csrRowPtr = nullptr;
        other.d_csrColInd = nullptr;
        other.d_csrVal = nullptr;
        other.d_csrVal_bf16 = nullptr;
    }
    
    SparseMatrixMixed& operator=(SparseMatrixMixed&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            cleanup();
            
            // Transfer ownership
            rows = other.rows;
            cols = other.cols;
            nnz = other.nnz;
            d_csrRowPtr = other.d_csrRowPtr;
            d_csrColInd = other.d_csrColInd;
            d_csrVal = other.d_csrVal;
            d_csrVal_bf16 = other.d_csrVal_bf16;
            
            // Reset other's pointers
            other.d_csrRowPtr = nullptr;
            other.d_csrColInd = nullptr;
            other.d_csrVal = nullptr;
            other.d_csrVal_bf16 = nullptr;
        }
        return *this;
    }
};

// Structure to hold only sparse matrix in CSR format
template<typename T>
struct SparseMatrix {
    int rows, cols, nnz;
    int *d_csrRowPtr, *d_csrColInd;
    T *d_csrVal;
    
    // Constructor
    SparseMatrix() : rows(0), cols(0), nnz(0), 
                    d_csrRowPtr(nullptr), d_csrColInd(nullptr), d_csrVal(nullptr) {}
    
    // Destructor
    ~SparseMatrix() {
        cleanup();
    }
    
    // Manual cleanup method - can be called explicitly if needed
    void cleanup() {
        if (d_csrRowPtr) {
            cudaFree(d_csrRowPtr);
            d_csrRowPtr = nullptr;
        }
        if (d_csrColInd) {
            cudaFree(d_csrColInd);
            d_csrColInd = nullptr;
        }
        if (d_csrVal) {
            cudaFree(d_csrVal);
            d_csrVal = nullptr;
        }
        rows = cols = nnz = 0;
    }
    
    // Check if matrix has been allocated
    bool isEmpty() const {
        return (d_csrRowPtr == nullptr && d_csrColInd == nullptr && d_csrVal == nullptr);
    }
    
    // Disable copy constructor and assignment operator to prevent double-free
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    
    // Enable move constructor and assignment operator
    SparseMatrix(SparseMatrix&& other) noexcept 
        : rows(other.rows), cols(other.cols), nnz(other.nnz),
          d_csrRowPtr(other.d_csrRowPtr), d_csrColInd(other.d_csrColInd),
          d_csrVal(other.d_csrVal) {
        other.d_csrRowPtr = nullptr;
        other.d_csrColInd = nullptr;
        other.d_csrVal = nullptr;
    }
    
    SparseMatrix& operator=(SparseMatrix&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            cleanup();
            
            // Transfer ownership
            rows = other.rows;
            cols = other.cols;
            nnz = other.nnz;
            d_csrRowPtr = other.d_csrRowPtr;
            d_csrColInd = other.d_csrColInd;
            d_csrVal = other.d_csrVal;
            
            // Reset other's pointers
            other.d_csrRowPtr = nullptr;
            other.d_csrColInd = nullptr;
            other.d_csrVal = nullptr;
        }
        return *this;
    }
};

// Function to read MTX file
bool readMTXFile(const char* filename, SparseMatrix<double> &matrix) {
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

// CUDA kernel to convert double to half precision
__global__ void double_to_bf16_kernel(double *d_double, __nv_bfloat16 *d_half, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_half[idx] = __float2bfloat16((float)d_double[idx]);
    }
}

// CUDA kernel to convert half to double precision
__global__ void bf16_to_double_kernel(__nv_bfloat16 *d_half, double *d_double, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_double[idx] = (double)__bfloat162float(d_half[idx]);
    }
}

// CG solver in half precision (using manual implementations)
int cg_half_precision(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle, 
                      SparseMatrix<__nv_bfloat16> &A, __nv_bfloat16 *d_b, __nv_bfloat16 *d_x, 
                      double tol = 1e-3, int max_iter = 10) {
    
    int n = A.rows;
    __nv_bfloat16 *d_r, *d_p, *d_Ap;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(__nv_bfloat16)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(__nv_bfloat16)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    float one_ht = 1.0f, zero_ht = 0.0f;

    // rsold = r^T * r
    __nv_bfloat16 rsold = 0.0f;
    float rsold_float = 0.0f;
    CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_16BF, 
                1,
                d_r, 
                CUDA_R_16BF,
                1, 
                &rsold, 
                CUDA_R_16BF, CUDA_R_32F));
    rsold_float = __bfloat162float(rsold);

    // debug
    // printf("Initial rsold: %f\n", rsold_float);
    // printf("Initial r (first 10 elements):\n");
    // std::vector<__nv_bfloat16> h_r(n);
    // CHECK_CUDA(cudaMemcpy(h_r.data(), d_r, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 10 && i < n; i++) {
    //     printf("%f ", __bfloat162float(h_r[i]));
    // }
    // printf("\n");


    // debug
    // print first 10 elements of p
    // std::vector<__nv_bfloat16> h_p(n);
    // CHECK_CUDA(cudaMemcpy(h_p.data(), d_p, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    // printf("Initial p (first 10 elements):\n");
    // for (int i = 0; i < 10 && i < n; i++) {
    //     printf("%f ", __bfloat162float(h_p[i]));
    // }
    // printf("\n");

    
    int iter, count = 0;

    __nv_bfloat16 pAp = 0.0f;
    __nv_bfloat16 dot_r = 0.0f;
    float pAp_float = 0.0f;
    float dot_r_float = 0.0f;

    float rsnew = 1.0f;
    float beta = 0.0f;
    float alpha1 = 0.0f;
    float neg_alpha = 0.0f;

    // Pre-allocate buffer for SpMV operations
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp;
    cusparseSpMatDescr_t A_mat_half;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_16BF));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_16BF));

    CHECK_CUSPARSE(cusparseCreateCsr(&A_mat_half, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF));

    // Get buffer size
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one_ht, A_mat_half, vecP, &zero_ht, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one_ht, A_mat_half, vecP, &zero_ht, vecAp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_p, 
                CUDA_R_16BF, 
                1,
                d_Ap, 
                CUDA_R_16BF,
                1, 
                &pAp, 
                CUDA_R_16BF, CUDA_R_32F));
        pAp_float = __bfloat162float(pAp);

        alpha1 = rsold_float / pAp_float;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &alpha1, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_16BF, 
                1, 
                d_x, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        
        // r = r - alpha * Ap
        neg_alpha = -alpha1;
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &neg_alpha, 
                CUDA_R_32F, 
                d_Ap, 
                CUDA_R_16BF, 
                1, 
                d_r, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_16BF, 
                1,
                d_r, 
                CUDA_R_16BF,
                1, 
                &dot_r, 
                CUDA_R_16BF, CUDA_R_32F));
        dot_r_float = __bfloat162float(dot_r);

        if (rsnew <= dot_r_float) {
           count++;
           rsnew = dot_r_float;
           if (count >= 2) {
                // printf("Early exit: rsnew <= dot_r, stopping CG iterations.\n");
                break;
            }
        }else {
            rsnew = dot_r_float;
            count = 0;
        }
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        beta = rsnew / rsold_float;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
                n, 
                &beta, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &one_ht, 
                CUDA_R_32F, 
                d_r, 
                CUDA_R_16BF, 
                1, 
                d_p, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        
        rsold_float = rsnew;

    }
    
    // Final cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_mat_half));
//    printf("CG half precision completed in %d iterations with residual %e\n", iter, sqrt(rsnew));

    
    return iter;
}

// CG solver in half precision (using manual implementations)
int cg_half_precision1(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle, 
                      SparseMatrix<__nv_bfloat16> &AIMS, SparseMatrix<__nv_bfloat16> &AIPS, __nv_bfloat16 *d_b, __nv_bfloat16 *d_x, 
                      double tol = 1e-3, int max_iter = 10) {
    
    int n = AIMS.rows;
    __nv_bfloat16 *d_r, *d_p, *d_Ap, *d_tmp;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemset(d_tmp, 0, n * sizeof(__nv_bfloat16)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(__nv_bfloat16)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    
    float one_ht = 1.0f, zero_ht = 0.0f;

    // rsold = r^T * r
    __nv_bfloat16 rsold = 0.0f;
    float rsold_float = 0.0f;
    CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_16BF, 
                1,
                d_r, 
                CUDA_R_16BF,
                1, 
                &rsold, 
                CUDA_R_16BF, CUDA_R_32F));
    rsold_float = __bfloat162float(rsold);

    // debug
    // printf("Initial rsold: %f\n", rsold_float);
    // printf("Initial r (first 10 elements):\n");
    // std::vector<__nv_bfloat16> h_r(n);
    // CHECK_CUDA(cudaMemcpy(h_r.data(), d_r, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 10 && i < n; i++) {
    //     printf("%f ", __bfloat162float(h_r[i]));
    // }
    // printf("\n");


    // debug
    // print first 10 elements of p
    // std::vector<__nv_bfloat16> h_p(n);
    // CHECK_CUDA(cudaMemcpy(h_p.data(), d_p, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    // printf("Initial p (first 10 elements):\n");
    // for (int i = 0; i < 10 && i < n; i++) {
    //     printf("%f ", __bfloat162float(h_p[i]));
    // }
    // printf("\n");

    
    int iter, count = 0;

    __nv_bfloat16 pAp = 0.0f;
    __nv_bfloat16 dot_r = 0.0f;
    float pAp_float = 0.0f;
    float dot_r_float = 0.0f;

    float rsnew = 1.0f;
    float beta = 0.0f;
    float alpha1 = 0.0f;
    float neg_alpha = 0.0f;

    // Pre-allocate buffer for SpMV operations
    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp, vecTmp;
    cusparseSpMatDescr_t AIPS_mat, AIMS_mat;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_16BF));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTmp, n, d_tmp, CUDA_R_16BF));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_16BF));

    CHECK_CUSPARSE(cusparseCreateCsr(&AIPS_mat, AIPS.rows, AIPS.cols, AIPS.nnz,
                                     AIPS.d_csrRowPtr, AIPS.d_csrColInd, AIPS.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF));

    CHECK_CUSPARSE(cusparseCreateCsr(&AIMS_mat, AIMS.rows, AIMS.cols, AIMS.nnz,
                                     AIMS.d_csrRowPtr, AIMS.d_csrColInd, AIMS.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF));

    // Get buffer size
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one_ht, AIPS_mat, vecP, &zero_ht, vecTmp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one_ht, AIMS_mat, vecTmp, &zero_ht, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one_ht, AIPS_mat, vecP, &zero_ht, vecTmp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one_ht, AIMS_mat, vecTmp, &zero_ht, vecAp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));

        // CHECK_CUDA(cudaMemset(d_tmp, 0, n * sizeof(__nv_bfloat16)));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_p, 
                CUDA_R_16BF, 
                1,
                d_Ap, 
                CUDA_R_16BF,
                1, 
                &pAp, 
                CUDA_R_16BF, CUDA_R_32F));
        pAp_float = __bfloat162float(pAp);

        alpha1 = rsold_float / pAp_float;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &alpha1, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_16BF, 
                1, 
                d_x, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        
        // r = r - alpha * Ap
        neg_alpha = -alpha1;
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &neg_alpha, 
                CUDA_R_32F, 
                d_Ap, 
                CUDA_R_16BF, 
                1, 
                d_r, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_16BF, 
                1,
                d_r, 
                CUDA_R_16BF,
                1, 
                &dot_r, 
                CUDA_R_16BF, CUDA_R_32F));
        dot_r_float = __bfloat162float(dot_r);

        if (rsnew <= dot_r_float) {
           count++;
           rsnew = dot_r_float;
           if (count >= 2) {
                // printf("Early exit: rsnew <= dot_r, stopping CG iterations.\n");
                break;
            }
        }else {
            rsnew = dot_r_float;
            count = 0;
        }
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        beta = rsnew / rsold_float;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
                n, 
                &beta, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &one_ht, 
                CUDA_R_32F, 
                d_r, 
                CUDA_R_16BF, 
                1, 
                d_p, 
                CUDA_R_16BF, 
                1, 
                CUDA_R_32F));
        
        rsold_float = rsnew;

    }
    
    // Final cleanup
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

// Half-precision iterative solver
int half_precision_solver(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                          SparseMatrix<double> &A, SparseMatrix<__nv_bfloat16> &AH, SparseMatrix<__nv_bfloat16> &AIMS, SparseMatrix<__nv_bfloat16> &AIPS, SparseMatrix<__nv_bfloat16> &S,
                          double *d_b, double *d_x0, double w, double alpha_param, double *res_out,
                          double tol = 1e-10, int max_iter = 5000) {
    
    int n = A.rows;
    double *d_r;
    __nv_bfloat16 *d_r_half, *d_r1_half, *d_temp_half;
    float alpha_param_f = (float)alpha_param;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r_half, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_r1_half, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_temp_half, n * sizeof(__nv_bfloat16)));
    
    double nr0;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_b, 1, &nr0));
    
    double res = 1.0;
    int kk = 0;
    long long total_cg_iters = 0;
    double one = 1.0, neg_one = -1.0;
    float one_f = 1.0f, neg_one_f = -1.0f, zero_f = 0.0f, factor;
    __nv_bfloat16 r3_norm;
    double r_norm;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cusparseDnVecDescr_t vecX0, vecR;
    cusparseDnVecDescr_t vecR1, vecTemp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX0, n, d_x0, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR1, n, d_r1_half, CUDA_R_16BF));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTemp, n, d_temp_half, CUDA_R_16BF));

    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    cusparseSpMatDescr_t matA, matS_half;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matS_half, S.rows, S.cols, S.nnz,
                                     S.d_csrRowPtr, S.d_csrColInd, S.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &neg_one, matA, vecX0, &one, vecR,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one_f, matS_half, vecR1, &zero_f, vecTemp,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    while (res > tol && kk < max_iter) {

        // r = b - A * x0        
        CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &neg_one, matA, vecX0, &one, vecR,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_r, 1, &r_norm));
        res = r_norm / nr0;
        
        // Convert r to half precision
        double_to_bf16_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_r_half, n);

        // debug
        // print first 10 of d_r_half
        // __nv_bfloat16 h_r_half[10];
        // CHECK_CUDA(cudaMemcpy(h_r_half, d_r_half, 10 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < 10; i++) {
        //     printf("d_r_half[%d] = %f\n", i, __bfloat162float(h_r_half[i]));
        // }
        // end debug

        // r1 = CG_solve(AH, r) in half precision
        total_cg_iters += cg_half_precision(cusparseHandle, cublasHandle, AH, d_r_half, d_r1_half);

        // debug
        // print first 10 of d_r1_half
        // __nv_bfloat16 h_r1_half[10];
        // CHECK_CUDA(cudaMemcpy(h_r1_half, d_r1_half, 10 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < 10; i++) {
        //     printf("d_r1_half[%d] = %f\n", i, __bfloat162float(h_r1_half[i]));
        // }
        // end debug

        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        // First: d_r1_half = alpha * I * d_r1_half = alpha * d_r1_half
        // CHECK_CUBLAS(cublasDscal(cublasHandle, n, &alpha_param, d_eye_r1, 1));
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
            n, 
            &alpha_param_f, 
            CUDA_R_32F, 
            d_r1_half, 
            CUDA_R_16BF, 
            1, 
            CUDA_R_32F));
        
        // Then: temp = S * r1
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one_f, matS_half, vecR1, &zero_f, vecTemp,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));

        
        // r2 = alpha * r1 - S * r1 = eye_r1 - temp
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
            n, 
            &neg_one_f, 
            CUDA_R_32F, 
            d_temp_half, 
            CUDA_R_16BF, 
            1, 
            d_r1_half, 
            CUDA_R_16BF, 
            1,
            CUDA_R_32F));
        
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        factor = (2.0 - w) * alpha_param;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
            n, 
            &factor, 
            CUDA_R_32F,
            d_r1_half, 
            CUDA_R_16BF,
            1,
            CUDA_R_32F));
        
        // r3 = CG_solve(ASS, r2) in half precision
        total_cg_iters += cg_half_precision1(cusparseHandle, cublasHandle, AIMS, AIPS, d_r1_half, d_r_half);
        
        // Check convergence
        CHECK_CUBLAS(cublasNrm2Ex(cublasHandle, 
            n, 
            d_r_half, 
            CUDA_R_16BF, 
            1, 
            &r3_norm,
            CUDA_R_16BF,
            CUDA_R_32F));
        if (__bfloat162float(r3_norm) == 0) {
            break;
        }

        // Convert r3 back to double precision
        bf16_to_double_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r_half, d_r, n);
        
        // x0 = x0 + r3
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_x0, 1));

        kk++;
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX0));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR1));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecTemp));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matS_half));
    if (dBuffer) cudaFree(dBuffer);
    if (dBuffer1) cudaFree(dBuffer1);
    
    printf("Half-precision solver: res = %e, iter = %d\n", res, kk);
    printf("Total CG iterations: %lld\n", total_cg_iters);
    
    // Cleanup
    cudaFree(d_r);
    cudaFree(d_r_half);
    cudaFree(d_r1_half);
    cudaFree(d_temp_half);

    *res_out = res;
    
    return kk;
}

// Function to create identity matrix in CSR format
void createIdentityMatrix(SparseMatrix<double> &I, int n) {
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
void transposeMatrix(cusparseHandle_t cusparseHandle, const SparseMatrix<double> &A, SparseMatrix<double> &AT) {
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
template<typename T>
struct needs_mixed_conversion : std::false_type {};

template<>
struct needs_mixed_conversion<SparseMatrixMixed> : std::true_type {};

template<typename T>
struct needs_bf16_conversion : std::false_type {};

template<>
struct needs_bf16_conversion<SparseMatrix<__nv_bfloat16>> : std::true_type {};

template<typename Type1, typename Type2, typename Type3>
void matrixAdd(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
               const Type1 &A, const Type2 &B, Type3 &C,
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
    if constexpr (needs_mixed_conversion<Type3>::value) {
        // If output type is mixed precision, allocate double precision storage
        CHECK_CUDA(cudaMalloc(&C.d_csrVal, C.nnz * sizeof(double)));
        // Also allocate bf16 precision storage
        CHECK_CUDA(cudaMalloc(&C.d_csrVal_bf16, C.nnz * sizeof(__nv_bfloat16)));
    } else if constexpr (needs_bf16_conversion<Type3>::value) {
        // If output type is bf16 precision, allocate bf16 precision storage
        CHECK_CUDA(cudaMalloc(&C.d_csrVal, C.nnz * sizeof(__nv_bfloat16)));
    } else {
        // Otherwise, allocate double precision storage
        CHECK_CUDA(cudaMalloc(&C.d_csrVal, C.nnz * sizeof(double)));
    }
    
    // Step 4: Perform the actual addition: C = scaleA * A + scaleB * B
    if constexpr (needs_bf16_conversion<Type3>::value) {
        // If output type is bf16 precision, use temporary double precision storage
        double *temp;
        CHECK_CUDA(cudaMalloc(&temp, C.nnz * sizeof(double)));
        CHECK_CUSPARSE(cusparseDcsrgeam2(cusparseHandle, C.rows, C.cols,
                                         &scaleA, descrA, A.nnz, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                         &scaleB, descrB, B.nnz, B.d_csrVal, B.d_csrRowPtr, B.d_csrColInd,
                                         descrC, temp, C.d_csrRowPtr, C.d_csrColInd, pBuffer));
        // Convert to float precision
        int threadsPerBlock = 256;
        int blocksPerGrid = (C.nnz + threadsPerBlock - 1) / threadsPerBlock;
        double_to_bf16_kernel<<<blocksPerGrid, threadsPerBlock>>>(temp, C.d_csrVal, C.nnz);
        CHECK_CUDA(cudaFree(temp));
    } else{
        CHECK_CUSPARSE(cusparseDcsrgeam2(cusparseHandle, C.rows, C.cols,
                                     &scaleA, descrA, A.nnz, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                     &scaleB, descrB, B.nnz, B.d_csrVal, B.d_csrRowPtr, B.d_csrColInd,
                                     descrC, C.d_csrVal, C.d_csrRowPtr, C.d_csrColInd, pBuffer));
    }

    // 如果输出类型需要bf16精度，分配额外的bf16精度存储
    if constexpr (needs_mixed_conversion<Type3>::value) {
        CHECK_CUDA(cudaMalloc(&C.d_csrVal_bf16, C.nnz * sizeof(__nv_bfloat16)));
        int threadsPerBlock = 256;
        int blocksPerGrid = (C.nnz + threadsPerBlock - 1) / threadsPerBlock;
        // Convert to bf16 precision
        double_to_bf16_kernel<<<blocksPerGrid, threadsPerBlock>>>(C.d_csrVal, C.d_csrVal_bf16, C.nnz);
    }
    
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
    SparseMatrix<double> A;
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
    SparseMatrix<double> I;
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
    SparseMatrix<double> AT;
    transposeMatrix(cusparseHandle, A, AT);
    
    // Step 2: Compute h = 0.5 * (A + A^T) - symmetric part
    SparseMatrix<double> H;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, H, 0.5, 0.5);

    // Step 3: Compute s = 0.5 * (A - A^T) - skew-symmetric part
    SparseMatrix<double> S_double;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, S_double, 0.5, -0.5);

    SparseMatrix<__nv_bfloat16> S;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, S, 0.5, -0.5);

    // Step 4: Compute AH = alpha * I + h
    SparseMatrix<__nv_bfloat16> AH;
    matrixAdd(cusparseHandle, cublasHandle, I, H, AH, alpha, 1.0);
    
    // Step 5: Compute asx = alpha * I + s
    SparseMatrix<__nv_bfloat16> ASX;
    matrixAdd(cusparseHandle, cublasHandle, I, S_double, ASX, alpha, 1.0);
    
    // Step 6: Compute alpha * I - s
    SparseMatrix<__nv_bfloat16> ALPHA_I_MINUS_S;
    matrixAdd(cusparseHandle, cublasHandle, I, S_double, ALPHA_I_MINUS_S, alpha, -1.0);
    
    printf("Matrix computation completed.\n");
    printf("H: %d x %d, nnz = %d\n", H.rows, H.cols, H.nnz);
    printf("S: %d x %d, nnz = %d\n", S.rows, S.cols, S.nnz);  
    printf("AH: %d x %d, nnz = %d\n", AH.rows, AH.cols, AH.nnz);

    AT.cleanup();   // Manually release transpose matrix
    H.cleanup();    // Manually release symmetric part
    S_double.cleanup(); // Manually release skew-symmetric part (double precision)
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

    // Run half-precision solver
    printf("\n=== Running Half-Precision Solver ===\n");
    
    cudaEvent_t start_half, stop_half;
    CHECK_CUDA(cudaEventCreate(&start_half));
    CHECK_CUDA(cudaEventCreate(&stop_half));
    
    CHECK_CUDA(cudaEventRecord(start_half));
    int iter_half = half_precision_solver(cusparseHandle, cublasHandle, A, AH, ALPHA_I_MINUS_S, ASX, S,
                                         d_b, d_x0, w, alpha, &res, 1e-10, 8000);
    // int iter_half = 0;
    CHECK_CUDA(cudaEventRecord(stop_half));
    CHECK_CUDA(cudaEventSynchronize(stop_half));
    
    float time_half;
    CHECK_CUDA(cudaEventElapsedTime(&time_half, start_half, stop_half));
    
    printf("Half-precision iterations: %d\n", iter_half);
    printf("Half-precision time: %.4f seconds\n", time_half / 1000.0f);
    
    printMemoryUsage();
    
    // Cleanup
    cudaFree(d_b);
    cudaFree(d_x0);
    
    // Cleanup original matrices
    
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    
    CHECK_CUDA(cudaEventDestroy(start_half));
    CHECK_CUDA(cudaEventDestroy(stop_half));
    
    return 0;
}
