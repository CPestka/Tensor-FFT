//Used for the creation of comparission data via CuFFT with varrying precissions
#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <optional>

#include "../../base/Plan.h"
#include "../../ComputeFFT.h"
#include "../DataMaker.cu"

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

std::unique_ptr<cufftDoubleComplex[]> GetComparisionFP64Data(
    float2* dptr_weights, int amount_of_frequencies, int fft_length,
    double normalization_factor){
  //Allocate device memory
  cufftDoubleComplex* dptr_data;
  cufftDoubleComplex* dptr_results;
  cudaMalloc(&dptr_data, 2 * sizeof(cufftDoubleComplex) * fft_length);
  dptr_results = dptr_data + fft_length;

  //Produce input data based on weights
  SineSupperposition<cufftDoubleComplex><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_data, dptr_weights, amount_of_frequencies,
      normalization_factor);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed.\n" << std::endl;
  }

  r = cufftPlanMany(&plan, 1, &fft_length, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_Z2Z, 1);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
  }

  r = cufftExecZ2Z(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan execution failed." << std::endl;
  }

  std::unique_ptr<cufftDoubleComplex[]> data =
      std::make_unique<cufftDoubleComplex[]>(fft_length);

  cudaMemcpy(data.get(), dptr_results, fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cufftDestroy(plan);
  cudaFree(dptr_data);

  return data;
}

std::unique_ptr<cufftComplex[]> GetComparisionFP32Data(
    float2* dptr_weights, int amount_of_frequencies, int fft_length,
    double normalization_factor){
  //Allocate device memory
  cufftComplex* dptr_data;
  cufftComplex* dptr_results;
  cudaMalloc(&dptr_data, 2 * sizeof(cufftComplex) * fft_length);
  dptr_results = dptr_data + fft_length;

  //Produce input data based on weights
  SineSupperposition<cufftComplex><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_data, dptr_weights, amount_of_frequencies,
      normalization_factor);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed.\n" << std::endl;
  }

  r = cufftPlanMany(&plan, 1, &fft_length, nullptr, 1, 1, nullptr, 1, 1,
                    CUFFT_C2C, 1);
  if (r != CUFFT_SUCCESS) {
    std::cout <<  "Error! Plan creation failed." << std::endl;
  }

  r = cufftExecC2C(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    std::cout <<  "Error! Plan execution failed." << std::endl;
  }

  std::unique_ptr<cufftComplex[]> data =
      std::make_unique<cufftComplex[]>(fft_length);

  cudaMemcpy(data.get(), dptr_results, fft_length * sizeof(cufftComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cufftDestroy(plan);
  cudaFree(dptr_data);

  return data;
}

std::unique_ptr<__half2[]> GetComparisionFP16Data(
    float2* dptr_weights, int amount_of_frequencies, long long fft_length,
    double normalization_factor){
  //Allocate device memory
  __half2* dptr_data;
  __half2* dptr_results;
  cudaMalloc(&dptr_data, 2 * sizeof(__half2) * fft_length);
  dptr_results = dptr_data + fft_length;

  //Produce input data based on weights
  SineSupperposition<__half2><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_data, dptr_weights, amount_of_frequencies,
      normalization_factor);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed.\n" << std::endl;
  }

  size_t size = 0;
  r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1, CUDA_C_16F,
                          nullptr, 1, 1, CUDA_C_16F, 1, &size, CUDA_C_16F);
  if (r != CUFFT_SUCCESS) {
    std::cout <<  "Error! Plan creation failed." << std::endl;
  }

  r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    std::cout <<  "Error! Plan execution failed." << std::endl;
  }

  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(fft_length);

  cudaMemcpy(data.get(), dptr_results, fft_length * sizeof(__half2),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cufftDestroy(plan);
  cudaFree(dptr_data);

  return data;
}

template<typename Integer>
std::unique_ptr<__half2[]> GetOurFP16Data(
    float2* dptr_weights, int amount_of_frequencies, long long fft_length,
    double normalization_factor){
  std::optional<Plan> possible_plan = MakePlan(fft_length);
  Plan my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
  }

  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    std::cout << "Error! Plan imcompatible with used device." << std::endl;
  }

  //Allocate device memory
  __half2* dptr_input_data = nullptr;
  __half2* dptr_output_data = nullptr;
  cudaMalloc(&dptr_input_data, 2 * sizeof(__half2) * fft_length);
  dptr_output_data = dptr_input_data + fft_length;

  //Produce input data based on weights
  SineSupperposition<__half2><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_input_data, dptr_weights, amount_of_frequencies,
      normalization_factor);

  //Compute the FFT on the device
  std::optional<std::string> error_mess =
      ComputeFFT<Integer>(my_plan, dptr_input_data, dptr_output_data,
                          GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
  }

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Allocate mem on host for results
  std::unique_ptr<__half2[]> results = std::make_unique<__half2[]>(fft_length);

  //Copy results back
  if (cudaMemcpy(results.get(),
                 my_plan.results_in_results_ ?
                     dptr_output_data : dptr_input_data,
                 fft_length * sizeof(__half2),
                 cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  return results;
}
