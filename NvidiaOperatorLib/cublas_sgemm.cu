#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

/// copy from
/// https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/sgemm/sgemm_cublas.cu
/// single precision(float32) GEMM
void cublas_sgemm(float *A, float *B, float *C, int M, int N, int K) {
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  static float alpha = 1.0;
  static float beta = 0.0;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);
}

int main(int argc, char **argv) {

  int M, K, N;
  M = K = N = 4096;
  if (argc == 4) {
    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N = atoi(argv[3]);
  }
  else{
    std::cerr << "Usage: " << argv[0] << " <M> <K> <N>" << std::endl;
    std::cerr << "Default GEMM size: 4096x4096x4096 with single precision(float32)" << std::endl;
  }

  float *h_A = (float *)malloc(M * K * sizeof(float));
  float *h_B = (float *)malloc(K * N * sizeof(float));
  float *h_C = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; i++)
    h_A[i] = 1.0;
  for (int i = 0; i < K * N; i++)
    h_B[i] = 1.0;

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, M * K * sizeof(float));
  cudaMalloc((void **)&d_B, K * N * sizeof(float));
  cudaMalloc((void **)&d_C, M * N * sizeof(float));

  // get cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);
  cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);

  // warmup
  std::cout << "Execute single precision(float32) GEMM with " << M << "x" << K << "x" << N << std::endl;
  cublas_sgemm(d_A, d_B, d_C, M, N, K);

  auto start = std::chrono::high_resolution_clock::now();
  const int run_times = 10;
  for (int i = 0; i < run_times; ++i) {
    cublas_sgemm(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
  }
  auto end = std::chrono::high_resolution_clock::now();

  cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Average time cost: "
            << duration.count() / static_cast<double>(run_times) << " ms"
            << std::endl;
  std::cout << static_cast<double>(2) * M * K * N * run_times /
                   (duration.count() * 1e9)
            << " TFLOPS" << std::endl;

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
