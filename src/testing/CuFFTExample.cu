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

  std::vector<float> weights_RE { 1.0, 0.7, 0.5, 0.2, 0.3, 0.7, 0.8 };
  std::vector<float> weights_IM { 1.0, 0.3, 0.2, 0.4, 0.9, 0.1, 0.6 };
  std::unique_ptr<cufftDoubleComplex[]> data =
      CreateSineSuperpostionDouble(fft_length, weights_RE, weights_IM);

  cufftDoubleComplex* dptr_data;
  cufftDoubleComplex* dptr_results;
  cudaMalloc(&dptr_data, sizeof(cufftDoubleComplex) * fft_length);
  cudaMalloc(&dptr_results, sizeof(cufftDoubleComplex) * fft_length);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  size_t size = 0;
  // r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1,
  //                         CUDA_C_16F, nullptr, 1, 1, CUDA_C_16F, 1, &size,
  //                         CUDA_C_16F);
  r = cufftPlan1d(&plan, fft_length, CUFFT_Z2Z, 1);

  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyHostToDevice);

  // r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  r = cufftExecC2C(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan execution failed." << std::endl;
    return false;
  }

  cudaMemcpy(data.get(), dptr_results, fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  //Write results to file
  WriteResultsToFileDouble2("example_results.dat", fft_length, data.get());

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  return true;
}
