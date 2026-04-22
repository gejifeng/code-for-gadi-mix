#include "fp64_gmres.h"

#include <cmath>
#include <cstdio>
#include <vector>

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

FP64GMRES::FP64GMRES(int n, int nnz, const CSRMatrix& A)
    : n(n), nnz(nnz), A(A), d_buffer_mv(nullptr), buffer_size_mv(0) {
    print_memory_usage("Start Constructor");
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

    CHECK_CUSPARSE(cusparseCreateCsr(&matA_descr, n, n, nnz,
                                     A.d_row_ptr, A.d_col_ind, A.d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    print_memory_usage("End Constructor");
}

FP64GMRES::~FP64GMRES() {
    if (d_buffer_mv) {
        CHECK_CUDA(cudaFree(d_buffer_mv));
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matA_descr));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
}

void FP64GMRES::solve(const double* d_b, double* d_x, int restart, int max_iters, double tol,
                      OrthogonalizationMethod method) {
    print_memory_usage("Start Solve");

    double* d_z;
    CHECK_CUDA(cudaMalloc(&d_z, (size_t)n * sizeof(double)));

    double* d_r;
    CHECK_CUDA(cudaMalloc(&d_r, (size_t)n * sizeof(double)));

    double* d_V;
    CHECK_CUDA(cudaMalloc(&d_V, (size_t)n * (restart + 1) * sizeof(double)));

    double* d_w;
    CHECK_CUDA(cudaMalloc(&d_w, (size_t)n * sizeof(double)));

    double* d_h_vec = nullptr;
    if (method == CGS) {
        CHECK_CUDA(cudaMalloc(&d_h_vec, (restart + 1) * sizeof(double)));
    }
    print_memory_usage("After Vector Allocations");

    std::vector<double> H((restart + 1) * restart, 0.0);
    std::vector<double> s(restart + 1, 0.0);
    std::vector<double> cs(restart, 0.0);
    std::vector<double> sn(restart, 0.0);

    cusparseDnVecDescr_t vec_z_descr;
    cusparseDnVecDescr_t vec_x_descr;
    cusparseDnVecDescr_t vec_b_descr;
    cusparseDnVecDescr_t vec_w_descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_z_descr, n, d_z, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x_descr, n, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_b_descr, n, (void*)d_b, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w_descr, n, d_w, CUDA_R_64F));

    double alpha_residual = -1.0;
    double beta_residual = 1.0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_residual, matA_descr, vec_x_descr, &beta_residual,
                                           vec_z_descr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buffer_size_mv));
    CHECK_CUDA(cudaMalloc(&d_buffer_mv, buffer_size_mv));
    print_memory_usage("After Buffer Allocations");

    double norm_b;
    CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, d_b, 1, &norm_b));
    if (norm_b == 0.0) {
        norm_b = 1.0;
    }

    printf("Initial Norm b: %e\n", norm_b);

    double alpha_mv = 1.0;
    double beta_mv = 0.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        CHECK_CUDA(cudaMemcpy(d_z, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_residual, matA_descr, vec_x_descr, &beta_residual,
                                    vec_z_descr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                    d_buffer_mv));

        double norm_z;
        CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, d_z, 1, &norm_z));
        printf("Iter %d, Residual Norm: %e\n", iter, norm_z);

        if (norm_z / norm_b < tol) {
            printf("Converged!\n");
            break;
        }

        double beta;
        CHECK_CUBLAS(cublasDcopy(cublas_handle, n, d_z, 1, d_r, 1));
        CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, d_r, 1, &beta));

        double inv_beta = 1.0 / beta;
        CHECK_CUBLAS(cublasDcopy(cublas_handle, n, d_r, 1, d_V, 1));
        CHECK_CUBLAS(cublasDscal(cublas_handle, n, &inv_beta, d_V, 1));

        s[0] = beta;
        for (int i = 1; i <= restart; ++i) {
            s[i] = 0.0;
        }

        int j = 0;
        for (; j < restart; ++j) {
            cusparseDnVecDescr_t vec_vj_descr;
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_vj_descr, n, d_V + (size_t)j * n, CUDA_R_64F));
            CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha_mv, matA_descr, vec_vj_descr, &beta_mv, vec_w_descr,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_mv));
            CHECK_CUSPARSE(cusparseDestroyDnVec(vec_vj_descr));

            if (method == MGS) {
                for (int i = 0; i <= j; ++i) {
                    double h_val;
                    CHECK_CUBLAS(cublasDdot(cublas_handle, n, d_w, 1, d_V + (size_t)i * n, 1, &h_val));
                    H[i + j * (restart + 1)] = h_val;
                    double neg_h = -h_val;
                    CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &neg_h, d_V + (size_t)i * n, 1, d_w, 1));
                }
            } else {
                double one = 1.0;
                double zero = 0.0;
                double minus_one = -1.0;

                CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_T, n, j + 1,
                                         &one, d_V, n, d_w, 1, &zero, d_h_vec, 1));
                CHECK_CUDA(cudaMemcpy(&H[j * (restart + 1)], d_h_vec,
                                      (j + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, j + 1,
                                         &minus_one, d_V, n, d_h_vec, 1, &one, d_w, 1));
            }

            double h_next;
            CHECK_CUBLAS(cublasDnrm2(cublas_handle, n, d_w, 1, &h_next));
            H[j + 1 + j * (restart + 1)] = h_next;

            if (j + 1 < restart + 1) {
                double inv_h = 1.0 / h_next;
                CHECK_CUBLAS(cublasDcopy(cublas_handle, n, d_w, 1, d_V + (size_t)(j + 1) * n, 1));
                CHECK_CUBLAS(cublasDscal(cublas_handle, n, &inv_h, d_V + (size_t)(j + 1) * n, 1));
            }

            for (int i = 0; i < j; ++i) {
                double temp = cs[i] * H[i + j * (restart + 1)] + sn[i] * H[i + 1 + j * (restart + 1)];
                H[i + 1 + j * (restart + 1)] = -sn[i] * H[i + j * (restart + 1)] +
                                               cs[i] * H[i + 1 + j * (restart + 1)];
                H[i + j * (restart + 1)] = temp;
            }

            double h_jj = H[j + j * (restart + 1)];
            double h_j1j = H[j + 1 + j * (restart + 1)];

            if (fabs(h_j1j) > 1e-20) {
                double t = sqrt(h_jj * h_jj + h_j1j * h_j1j);
                cs[j] = h_jj / t;
                sn[j] = h_j1j / t;

                H[j + j * (restart + 1)] = cs[j] * h_jj + sn[j] * h_j1j;
                H[j + 1 + j * (restart + 1)] = 0.0;

                double s_j = s[j];
                s[j] = cs[j] * s_j;
                s[j + 1] = -sn[j] * s_j;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }
        }

        std::vector<double> y(j);
        for (int i = j - 1; i >= 0; --i) {
            y[i] = s[i];
            for (int k = i + 1; k < j; ++k) {
                y[i] -= H[i + k * (restart + 1)] * y[k];
            }
            y[i] /= H[i + i * (restart + 1)];
        }

        CHECK_CUDA(cudaMemset(d_r, 0, (size_t)n * sizeof(double)));
        for (int i = 0; i < j; ++i) {
            CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &y[i], d_V + (size_t)i * n, 1, d_r, 1));
        }

        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublas_handle, n, &one, d_r, 1, d_x, 1));
    }

    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_z_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_b_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_w_descr));

    if (d_h_vec) {
        CHECK_CUDA(cudaFree(d_h_vec));
    }
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_w));
}