//Used to profile the function ComputeFFT
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

int main(){
  long long fft_length = 16*16*16;

  std::vector<float> weights_RE { 1.0, 0.7, 0.5, 0.2, 1.0, 0.9, 0.4 };
  std::vector<float> weights_IM { 1.0, 0.3, 0.2, 0.4, 0.1, 0.5, 0.9 };
  std::unique_ptr<__half2[]> data =
      CreateSineSuperpostionH2(fft_length, weights_RE, weights_IM);

  __half2* dptr_data;
  __half2* dptr_results;
  cudaMalloc(&dptr_data, sizeof(__half2) * fft_length);
  cudaMalloc(&dptr_results, sizeof(__half2) * fft_length);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  size_t size = 0;
  r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1,
                          CUDA_C_16F, nullptr, 1, 1, CUDA_C_16F, 1, &size,
                          CUDA_C_16F);
  //r = cufftPlan1d(&plan, fft_length, CUFFT_C2C, 0);

  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(__half2),
             cudaMemcpyHostToDevice);

  r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  //r = cufftExecC2C(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan execution failed." << std::endl;
    return false;
  }

  cudaDeviceSynchronize();

  //Write results to file
  WriteResultsToFileHalf2("example_results.dat", fft_length, data.get());

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  return true;
}
