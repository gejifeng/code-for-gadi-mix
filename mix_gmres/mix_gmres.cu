#include "mix_gmres.h"
#include <vector>
#include <cmath>
#include <cstdio>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            printf("cuSPARSE error at %s %d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s %d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

void print_memory_usage(const char* stage) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
    if (err != cudaSuccess) {
        printf("Error getting memory info: %s\n", cudaGetErrorString(err));
    } else {
        double used_mb = (double)(total_byte - free_byte) / 1024.0 / 1024.0;
        double total_mb = (double)total_byte / 1024.0 / 1024.0;
        printf("[%s] GPU Memory Used: %.2f MB / %.2f MB\n", stage, used_mb, total_mb);
    }
}

// Kernel to cast double to float
__global__ void cast_double_to_float(int n, const double* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)src[idx];
    }
}

// Kernel to cast float to double and add: dst = dst + (double)src
__global__ void add_float_to_double(int n, const float* src, double* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += (double)src[idx];
    }
}

MixedPrecisionGMRES::MixedPrecisionGMRES(int n, int nnz, const CSRMatrix& A) 
    : n(n), nnz(nnz), A_high(A), d_val_low(nullptr), d_buffer_mv_high(nullptr), d_buffer_mv_low(nullptr) {
    
    print_memory_usage("Start Constructor");
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

    // Create High Precision Matrix Descriptor
    CHECK_CUSPARSE(cusparseCreateCsr(&matA_high_descr, n, n, nnz,
                                     A.d_row_ptr, A.d_col_ind, A.d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    convert_matrix_to_float();
    print_memory_usage("End Constructor");
}

MixedPrecisionGMRES::~MixedPrecisionGMRES() {
    if (d_val_low) CHECK_CUDA(cudaFree(d_val_low));
    if (d_buffer_mv_high) CHECK_CUDA(cudaFree(d_buffer_mv_high));
    if (d_buffer_mv_low) CHECK_CUDA(cudaFree(d_buffer_mv_low));
    
    CHECK_CUSPARSE(cusparseDestroySpMat(matA_high_descr));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA_low_descr));
    
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
}

void MixedPrecisionGMRES::convert_matrix_to_float() {
    CHECK_CUDA(cudaMalloc(&d_val_low, (size_t)nnz * sizeof(float)));
    
    // Launch kernel to convert values
    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;
    cast_double_to_float<<<numBlocks, blockSize>>>(nnz, A_high.d_val, d_val_low);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Create Low Precision Matrix Descriptor
    // Reuse row_ptr and col_ind from high precision matrix
    CHECK_CUSPARSE(cusparseCreateCsr(&matA_low_descr, n, n, nnz,
                                     A_high.d_row_ptr, A_high.d_col_ind, d_val_low,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
}

void MixedPrecisionGMRES::solve(const double* d_b, double* d_x, int restart, int max_iters, double tol, OrthogonalizationMethod method) {
    print_memory_usage("Start Solve");
    // Allocations
    double *d_z; // Residual in double
    CHECK_CUDA(cudaMalloc(&d_z, (size_t)n * sizeof(double)));
    
    float *d_r; // Residual in float
    CHECK_CUDA(cudaMalloc(&d_r, (size_t)n * sizeof(float)));
    
    float *d_V; // Arnoldi basis vectors (n x (restart+1))
    CHECK_CUDA(cudaMalloc(&d_V, (size_t)n * (restart + 1) * sizeof(float)));
    
    float *d_w; // Temporary vector
    CHECK_CUDA(cudaMalloc(&d_w, (size_t)n * sizeof(float)));

    // Buffer for CGS coefficients on device
    float *d_h_vec = nullptr;
    if (method == CGS) {
        CHECK_CUDA(cudaMalloc(&d_h_vec, (restart + 1) * sizeof(float)));
    }
    print_memory_usage("After Vector Allocations");

    // Host side Arnoldi variables
    std::vector<float> H((restart + 1) * restart, 0.0f); // Hessenberg matrix
    std::vector<float> s(restart + 1, 0.0f); // RHS for least squares
    std::vector<float> cs(restart, 0.0f); // Cosine of rotations
    std::vector<float> sn(restart, 0.0f); // Sine of rotations

    // Descriptors for SpMV
    cusparseDnVecDescr_t vec_z_descr, vec_x_descr, vec_b_descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_z_descr, n, d_z, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x_descr, n, d_x, CUDA_R_64F));
    // Note: d_b is const, but API takes non-const. We won't modify it.
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_b_descr, n, (void*)d_b, CUDA_R_64F));

    cusparseDnVecDescr_t vec_w_descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w_descr, n, d_w, CUDA_R_32F));
    
    // Buffer size queries
    double alpha_high = -1.0;
    double beta_high = 1.0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_high, matA_high_descr, vec_x_descr, &beta_high, vec_z_descr,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_mv_high));
    CHECK_CUDA(cudaMalloc(&d_buffer_mv_high, buffer_size_mv_high));

    float alpha_low = 1.0f;
    float beta_low = 0.0f;
    // We need a dummy descriptor for V[j] to query buffer size, or just query once with d_w
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_low, matA_low_descr, vec_w_descr, &beta_low, vec_w_descr,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_mv_low));
    CHECK_CUDA(cudaMalloc(&d_buffer_mv_low, buffer_size_mv_low));
    print_memory_usage("After Buffer Allocations");

    // Initial residual norm
    double norm_b;
    CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, d_b, 1, &norm_b));
    if (norm_b == 0.0) norm_b = 1.0;

    printf("Initial Norm b: %e\n", norm_b);

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. z_k = b - A * x_k (High Precision)
        // Copy b to z
        CHECK_CUDA(cudaMemcpy(d_z, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
        // z = z - A * x (alpha=-1, beta=1)
        CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_high, matA_high_descr, vec_x_descr, &beta_high, vec_z_descr,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_mv_high));
        
        double norm_z;
        CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, d_z, 1, &norm_z));
        
        printf("Iter %d, Residual Norm: %e\n", iter, norm_z);

        if (norm_z / norm_b < tol) {
            printf("Converged!\n");
            break;
        }

        // 2. r_k = cast(z_k) (Low Precision)
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        cast_double_to_float<<<numBlocks, blockSize>>>(n, d_z, d_r);
        
        // 3. Inner GMRES
        float beta;
        CHECK_CUBLAS(cublasSnrm2(cublas_handle, n, d_r, 1, &beta));
        
        // V[0] = r / beta
        float inv_beta = 1.0f / beta;
        CHECK_CUBLAS(cublasScopy(cublas_handle, n, d_r, 1, d_V, 1));
        CHECK_CUBLAS(cublasSscal(cublas_handle, n, &inv_beta, d_V, 1));
        
        s[0] = beta;
        for(int i=1; i<=restart; ++i) s[i] = 0.0f;

        int j = 0;
        for (; j < restart; ++j) {
            // w = A * V[j]
            // Create descriptor for V[j]
            cusparseDnVecDescr_t vec_vj_descr;
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_vj_descr, n, d_V + (size_t)j * n, CUDA_R_32F));
            
            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha_low, matA_low_descr, vec_vj_descr, &beta_low, vec_w_descr,
                                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_mv_low));
            
            CHECK_CUSPARSE(cusparseDestroyDnVec(vec_vj_descr));

            // Orthogonalization
            if (method == MGS) {
                // MGS
                for (int i = 0; i <= j; ++i) {
                    float h_val;
                    CHECK_CUBLAS(cublasSdot(cublas_handle, n, d_w, 1, d_V + (size_t)i * n, 1, &h_val));
                    H[i + j * (restart + 1)] = h_val;
                    float neg_h = -h_val;
                    CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &neg_h, d_V + (size_t)i * n, 1, d_w, 1));
                }
            } else {
                // CGS
                // h = V_{0:j}^T * w
                float one = 1.0f;
                float zero = 0.0f;
                float minus_one = -1.0f;
                
                // d_V is n x (restart+1), but we only use first j+1 columns
                // h_vec is (j+1) x 1
                CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, n, j + 1, 
                                         &one, d_V, n, d_w, 1, &zero, d_h_vec, 1));
                
                // Copy h to host H matrix
                CHECK_CUDA(cudaMemcpy(&H[0 + j * (restart + 1)], d_h_vec, (j + 1) * sizeof(float), cudaMemcpyDeviceToHost));
                
                // w = w - V_{0:j} * h
                CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_N, n, j + 1, 
                                         &minus_one, d_V, n, d_h_vec, 1, &one, d_w, 1));
            }
            
            float h_next;
            CHECK_CUBLAS(cublasSnrm2(cublas_handle, n, d_w, 1, &h_next));
            H[j + 1 + j * (restart + 1)] = h_next;
            
            // V[j+1] = w / h_next
            if (j + 1 < restart + 1) { 
                 float inv_h = 1.0f / h_next;
                 CHECK_CUBLAS(cublasScopy(cublas_handle, n, d_w, 1, d_V + (size_t)(j + 1) * n, 1));
                 CHECK_CUBLAS(cublasSscal(cublas_handle, n, &inv_h, d_V + (size_t)(j + 1) * n, 1));
            }

            // Apply Givens rotations to new column of H
            for (int i = 0; i < j; ++i) {
                float temp = cs[i] * H[i + j * (restart + 1)] + sn[i] * H[i + 1 + j * (restart + 1)];
                H[i + 1 + j * (restart + 1)] = -sn[i] * H[i + j * (restart + 1)] + cs[i] * H[i + 1 + j * (restart + 1)];
                H[i + j * (restart + 1)] = temp;
            }
            
            // Compute new rotation
            float h_jj = H[j + j * (restart + 1)];
            float h_j1j = H[j + 1 + j * (restart + 1)];
            
            if (fabs(h_j1j) > 1e-10) { 
                float t = sqrtf(h_jj * h_jj + h_j1j * h_j1j);
                cs[j] = h_jj / t;
                sn[j] = h_j1j / t;
                
                H[j + j * (restart + 1)] = cs[j] * h_jj + sn[j] * h_j1j;
                H[j + 1 + j * (restart + 1)] = 0.0f;
                
                // Apply to s
                float s_j = s[j];
                s[j] = cs[j] * s_j;
                s[j + 1] = -sn[j] * s_j;
            } else {
                cs[j] = 1.0f;
                sn[j] = 0.0f;
            }
            
            // Check inner convergence (optional)
            // if (fabs(s[j+1]) < tol * norm_b) { j++; break; }
        }
        
        // Solve upper triangular system H * y = s
        std::vector<float> y(j);
        for (int i = j - 1; i >= 0; --i) {
            y[i] = s[i];
            for (int k = i + 1; k < j; ++k) {
                y[i] -= H[i + k * (restart + 1)] * y[k];
            }
            y[i] /= H[i + i * (restart + 1)];
        }
        
        // Compute update u = V * y
        CHECK_CUDA(cudaMemset(d_r, 0, (size_t)n * sizeof(float)));
        for (int i = 0; i < j; ++i) {
            CHECK_CUBLAS(cublasSaxpy(cublas_handle, n, &y[i], d_V + (size_t)i * n, 1, d_r, 1));
        }
        
        // Update x = x + cast(u)
        add_float_to_double<<<numBlocks, blockSize>>>(n, d_r, d_x);
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_z_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_b_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_w_descr));
    
    if (d_h_vec) CHECK_CUDA(cudaFree(d_h_vec));
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_w));
}
