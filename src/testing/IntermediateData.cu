//Shows via a simple example how to compute FFTs via the provided functions
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.h"
#include "FileWriter.h"
#include "../base/TensorFFT4096_2.cu"

int main(){
  constexpr int fft_length = 16*16*16;

  // std::vector<float> weights_RE { 1.0, 0.7, 0.5, 0.2, 0.3, 0.7, 0.8 };
  // std::vector<float> weights_IM { 1.0, 0.3, 0.2, 0.4, 0.9, 0.1, 0.6 };

  std::vector<float> weights_RE { 1.0 };
  std::vector<float> weights_IM { 0.0 };
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionHGPU(fft_length, weights_RE, weights_IM, 7);

  WriteResultsToFile("Test_in_old.dat", fft_length, data.get());

  __half* dptr_in_RE;
  __half* dptr_in_IM;
  __half* dptr_out_RE;
  __half* dptr_out_IM;

  cudaMalloc((void*)&dptr_in_RE; sizeof(__half) * 4 * fft_length);
  dptr_in_IM = dptr_in_RE + fft_length;
  dptr_out_RE = dptr_in_IM + fft_length;
  dptr_out_IM = dptr_out_RE + fft_length;

  cudaMemcpy(dptr_in_RE, data, 2 * fft_length * sizeof(__half),
             cudaMemcpyHostToDevice);

  TensorFFT4096_2<<<1,512,32768>>>(dptr_in_RE, dptr_in_IM, dptr_out_RE,
                                   dptr_out_IM, fft_length, 2, 0);

  cudaDeviceSynchronize();

  cudaMemcpy(data, dptr_out_RE, 2 * fft_length * sizeof(__half),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  WriteResultsToFile("Test_out_old.dat", fft_length, data.get());
}
