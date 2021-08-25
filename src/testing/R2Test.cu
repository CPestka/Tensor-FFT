#include <iostream>
#include <vector>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.h"
#include "FileWriter.h"
#include "../base/Radix2.cu"

int main(){
  constexpr int fft_length = 2;
  std::vector<float> weights_RE { 1.0 };
  std::vector<float> weights_IM { 0.0 };
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostion(fft_length, weights_RE, weights_IM);

  data[1] = 0.5;

  WriteResultsToFile("r2_in.dat", fft_length, data.get());

  __half* dptr_data;
  cudaMalloc(&dptr_data, sizeof(__half) * 4 * fft_length);

  cudaMemcpy(dptr_data, data.get(),
             sizeof(__half) * 2 * fft_length, cudaMemcpyHostToDevice);

  __half* data_RE = dptr_data;
  __half* data_IM = dptr_data + fft_length;
  __half* results_RE = dptr_data + (2 * fft_length);
  __half* results_IM = dptr_data + (3 * fft_length);

  Radix2Kernel<<<1, fft_length / 2>>>(
      data_RE, data_IM, results_RE, results_IM, 1);



  cudaMemcpy(data.get(), results_RE,
             sizeof(__half) * 2 * fft_length, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  WriteResultsToFile("r2_out.dat", fft_length, data.get());
}
