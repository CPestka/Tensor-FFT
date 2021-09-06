#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../base/Transposer.cu"


int main() {
  int fft_length = 16*16*16*16*16 * 16*2;
  // __half2* dptr_data;
  // __half2* dptr_results;
  //
  // cudaMalloc(&dptr_data, 2 * sizeof(__half2) * fft_length);
  // dptr_results = dptr_data + fft_length;
  //
  // TransposeKernel<<<fft_length/512, 512>>>(
  //     dptr_data, dptr_results, fft_length, 5, 1);

  __half* dptr_data_RE;
  __half* dptr_results_RE;
  __half* dptr_data_IM;
  __half* dptr_results_IM;

  cudaMalloc(&dptr_data_RE, 4 * sizeof(__half) * fft_length);
  dptr_data_IM = dptr_data_RE + fft_length;
  dptr_results_RE = dptr_data_IM + fft_length;
  dptr_results_IM = dptr_results_RE + fft_length;

  TransposeKernel<<<fft_length/512, 512>>>(
      dptr_data_RE, dptr_data_IM, dptr_results_RE, dptr_results_IM,  fft_length, 5, 1);
}
