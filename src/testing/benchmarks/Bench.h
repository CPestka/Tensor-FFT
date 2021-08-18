//Contains various overloads of functions that benchmark the fft implementation
//given a fft_length, warmup_samples count and sample_size.
#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>
#include <cassert>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../TestingDataCreation.h"
#include "../Timer.h"
#include "../../base/ComputeFFT.h"
#include "../../base/Plan.h"
#include "BenchUtil.h"

struct BenchResult;

//Takes performance parameters as input
//Mostly intended for tuner.cu
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<BenchResult> Benchmark(const Integer fft_length,
                                     const int warmup_samples,
                                     const int sample_size,
                                     const BaseFFTMode mode,
                                     const int base_fft_warps_per_block,
                                     const int r16_warps_per_block,
                                     const int r2_blocksize){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostion(fft_length.back(),  weights);

  std::vector<double> runtime;

  std::optional<Plan> possible_plan =
      CreatePlan(fft_length, mode, base_fft_warps_per_block,
                 r16_warps_per_block, r2_blocksize);
  Plan my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  int device_id;
  assert((PlanWorksOnDevice(my_plan, cudaGetDevice(&device_id))));

  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

//Reads optimized parameters from file created by tuner.cu
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<BenchResult> Benchmark(const Integer fft_length,
                                     const int warmup_samples,
                                     const int sample_size,
                                     const std::string tuner_results_file){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostion(fft_length.back(),  weights);

  std::vector<double> runtime;

  std::optional<Plan> possible_plan =
      CreatePlan(fft_length, tuner_results_file);
  Plan my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  int device_id;
  assert((PlanWorksOnDevice(my_plan, cudaGetDevice(&device_id))));

  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

//Async versions of the above

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<BenchResult> Benchmark(const Integer fft_length,
                                     const int warmup_samples,
                                     const int sample_size,
                                     const int async_batch_size,
                                     const BaseFFTMode mode,
                                     const int base_fft_warps_per_block,
                                     const int r16_warps_per_block,
                                     const int r2_blocksize){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionBatch(fft_length, batch_size, weights);

  std::vector<double> runtime;

  std::optional<Plan> possible_plan =
      CreatePlan(fft_length, mode, base_fft_warps_per_block,
                 r16_warps_per_block, r2_blocksize);
  Plan my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  int device_id;
  assert((PlanWorksOnDevice(my_plan, cudaGetDevice(&device_id))));

  DataBatchHandler my_handler(fft_length, async_batch_size);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<BenchResult> Benchmark(const Integer fft_length,
                                     const int warmup_samples,
                                     const int sample_size,
                                     const int async_batch_size,
                                     const std::string tuner_results_file){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionBatch(fft_length, batch_size, weights);

  std::vector<double> runtime;

  std::optional<Plan> possible_plan =
      CreatePlan(fft_length, tuner_results_file);
  Plan my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  int device_id;
  assert((PlanWorksOnDevice(my_plan, cudaGetDevice(&device_id))));

  DataBatchHandler my_handler(fft_length, async_batch_size);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

//Bench for single fft of cufft
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<BenchResult> BenchmarkCuFFT(const Integer fft_length,
                                          const int warmup_samples,
                                          const int sample_size){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half2[]> data =
      CreateSineSuperpostionH2(fft_length,  weights);

  std::vector<double> runtime;

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
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(__half2),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    IntervallTimer computation_time;

    r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan execution failed." << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<BenchResult> BenchmarkCuFFT(const Integer fft_length,
                                          const int warmup_samples,
                                          const int sample_size,
                                          const int async_batch_size){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half2[]> data =
      CreateSineSuperpostionH2Batch(fft_length,  weights, batch_size);

  std::vector<double> runtime;

  __half2* dptr_data;
  __half2* dptr_results;
  cudaMalloc(&dptr_data, sizeof(__half2) * fft_length * batch_size);
  cudaMalloc(&dptr_results, sizeof(__half2) * fft_length * batch_size);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  size_t size = 0;
  r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1,
                          CUDA_C_16F, nullptr, 1, 1, CUDA_C_16F, batch_size,
                          &size, CUDA_C_16F);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    cudaMemcpy(dptr_data, data.get(),
               fft_length * sizeof(__half2) * batch_size,
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    IntervallTimer computation_time;

    r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan execution failed." << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}
