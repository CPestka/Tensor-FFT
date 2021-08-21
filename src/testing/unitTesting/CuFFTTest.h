#pragma once
//Provides comparision for the fft test via cuFFT

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#include "../FileWriter.h"
#include "../TestingDataCreation.h"

std::optional<std::string> CreateComparisonDataHalf(
    long long fft_length,
    const std::string file_name){
  std::vector<float> weights_RE{ 1.0, 0.7 };
  std::vector<float> weights_IM{ 0.0, 0.0 };
  std::unique_ptr<__half2[]> data =
      CreateSineSuperpostionH2(fft_length, weights_RE, weights_IM);

  __half2* dptr_data;
  __half2* dptr_results;
  cudaMalloc(&dptr_data, sizeof(__half2) * fft_length);
  cudaMalloc(&dptr_results, sizeof(__half2) * fft_length);
  cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(__half2),
             cudaMemcpyHostToDevice);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    return "Error! Plan creation failed.";
  }

  size_t size = 0;
  r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1, CUDA_C_16F,
                          nullptr, 1, 1, CUDA_C_16F, 1, &size, CUDA_C_16F);
  if (r != CUFFT_SUCCESS) {
    return "Error! Plan creation failed.";
  }

  r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    return "Error! Plan execution failed.";
  }

  cudaMemcpy(data.get(), dptr_results, fft_length * sizeof(__half2),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  WriteResultsToFileHalf2(file_name, fft_length, data.get());

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  return std::nullopt;
}

std::optional<std::string> CreateComparisonDataDouble(
    int fft_length,
    const std::string file_name){
  std::vector<float> weights_RE{ 1.0, 0.7 };
  std::vector<float> weights_IM{ 0.0, 0.0 };
  std::unique_ptr<cufftDoubleComplex[]> data =
      CreateSineSuperpostionDouble(fft_length, weights_RE, weights_IM);

  cufftDoubleComplex* dptr_data;
  cufftDoubleComplex* dptr_results;
  cudaMalloc(&dptr_data, sizeof(cufftDoubleComplex) * fft_length);
  cudaMalloc(&dptr_results, sizeof(cufftDoubleComplex) * fft_length);
  cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyHostToDevice);

  cufftHandle plan;
  cufftResult r;

  r = cufftPlanMany(&plan, 1, &fft_length, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_Z2Z, 1);
  if (r != CUFFT_SUCCESS) {
    return "Error! Plan creation failed.";
  }

  r = cufftExecZ2Z(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    return "Error! Plan execution failed.";
  }

  cudaMemcpy(data.get(), dptr_results, fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  WriteResultsToFileDouble2(file_name, fft_length, data.get());

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  return std::nullopt;
}
