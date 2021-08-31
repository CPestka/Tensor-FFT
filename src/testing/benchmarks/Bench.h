//Contains various overloads of functions that benchmark the fft implementation
//given a fft_length, warmup_samples count and sample_size.
#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../TestingDataCreation.h"
#include "../Timer.h"
#include "../../base/ComputeFFT.h"
#include "../../base/Plan.h"
#include "BenchUtil.h"

template <typename Integer>
std::vector<RunResults> RunBenchOverSearchSpace(
    const std::vector<RunConfig> &configs,
    const int sample_size,
    const int warmup_samples,
    const Integer fft_length){
  std::vector<RunResults> results;

  for(int i=0; i<static_cast<int>(configs.size()); i++){
    //Run Bench with a configuration
    std::optional<RunResults> possible_bench_result = Benchmark(
        fft_length, warmup_samples, sample_size, configs[i].mode_,
        configs[i].base_fft_warps_per_block_,
        configs[i].r16_warps_per_block_, configs[i].r2_blocksize_);

    //If Benchmark completed successfully save results.
    if (possible_bench_result) {
      results.push_back(possible_bench_result.value());
    }
  }

  return results;
}

template <typename Integer>
std::vector<RunResults> RunBenchOverSearchSpace(
    const std::vector<RunConfig> &configs,
    const int sample_size,
    const int warmup_samples,
    const int async_batch_size,
    const Integer fft_length){
  std::vector<RunResults> results;

  for(int i=0; i<static_cast<int>(configs.size()); i++){
    //Run Bench with a configuration
    std::optional<RunResults> possible_bench_result = Benchmark(
        fft_length, warmup_samples, sample_size, async_batch_size,
        configs[i].mode_, configs[i].base_fft_warps_per_block_,
        configs[i].r16_warps_per_block_, configs[i].r2_blocksize_);

    //If Benchmark completed successfully save results.
    if (possible_bench_result) {
      results.push_back(possible_bench_result.value());
    }
  }

  return results;
}


//Takes performance parameters as input
//Mostly intended for tuner.cu
template <typename Integer>
std::optional<RunResults> Benchmark(const Integer fft_length,
                                    const int warmup_samples,
                                    const int sample_size,
                                    const BaseFFTMode mode,
                                    const int base_fft_warps_per_block,
                                    const int r16_warps_per_block,
                                    const int r2_blocksize){
  //std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights_RE = GetRandomWeights(10, 42);
  std::vector<float> weights_IM = GetRandomWeights(10, 4242);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionHGPU(fft_length,  weights_RE, weights_IM, 10);

  std::vector<double> runtime;

  std::optional<Plan<Integer>> possible_plan =
      CreatePlan(fft_length, mode, base_fft_warps_per_block,
                 r16_warps_per_block, r2_blocksize);
  Plan<Integer> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return std::nullopt;
  }

  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    std::cout << "Plan doesnt work on device -> configuration skiped."
              << std::endl;
    return std::nullopt;
  }

  int max_no_optin_shared_mem = GetMaxNoOptInSharedMem(device_id);

  std::optional<std::string> error_mess;

  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return std::nullopt;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler, max_no_optin_shared_mem);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);
  RunResults tmp = {mode, base_fft_warps_per_block, r16_warps_per_block,
                    r2_blocksize, results};
  return tmp;
}

//Reads optimized parameters from file created by tuner.cu
template <typename Integer>
std::optional<RunResults> Benchmark(const Integer fft_length,
                                    const int warmup_samples,
                                    const int sample_size,
                                    const std::string tuner_results_file){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights_RE = GetRandomWeights(10, 42);
  std::vector<float> weights_IM = GetRandomWeights(10, 4242);

  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionHGPU(fft_length,  weights_RE, weights_IM, 10);

  std::vector<double> runtime;

  std::optional<Plan<Integer>> possible_plan =
      CreatePlan(fft_length, tuner_results_file);
  Plan<Integer> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return std::nullopt;
  }

  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    std::cout << "Plan doesnt work on device -> configuration skiped."
              << std::endl;
    return std::nullopt;
  }

  int max_no_optin_shared_mem = GetMaxNoOptInSharedMem(device_id);

  std::optional<std::string> error_mess;

  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return std::nullopt;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler, max_no_optin_shared_mem);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  RunResults tmp = {my_plan.base_fft_mode_, my_plan.base_fft_warps_per_block_,
                    my_plan.r16_warps_per_block_, my_plan.r2_blocksize_,
                    results};
  return tmp;
}

// //Async versions of the above
// template <typename Integer>
// std::optional<RunResults> Benchmark(const Integer fft_length,
//                                     const int warmup_samples,
//                                     const int sample_size,
//                                     const int async_batch_size,
//                                     const BaseFFTMode mode,
//                                     const int base_fft_warps_per_block,
//                                     const int r16_warps_per_block,
//                                     const int r2_blocksize){
//   std::cout << "Benchmarking fft_length: " << fft_length << std::endl;
//
//   std::vector<float> weights_RE = GetRandomWeights(10, 42);
//   std::vector<float> weights_IM = GetRandomWeights(10, 4242);
//   std::unique_ptr<__half[]> data =
//       CreateSineSuperpostionBatch(fft_length, async_batch_size,
//                                   weights_RE, weights_IM);
//
//   std::vector<double> runtime;
//
//   std::optional<Plan<Integer>> possible_plan =
//       CreatePlan(fft_length, mode, base_fft_warps_per_block,
//                  r16_warps_per_block, r2_blocksize);
//   Plan<Integer> my_plan;
//   if (possible_plan) {
//     my_plan = possible_plan.value();
//   } else {
//     std::cout << "Plan creation failed" << std::endl;
//     return std::nullopt;
//   }
//
//   int device_id;
//   cudaGetDevice(&device_id);
//   if (!PlanWorksOnDevice(my_plan, device_id)) {
//     std::cout << "Plan doesnt work on device -> configuration skiped."
//     return std::nullopt;
//   }
//
//   int max_no_optin_shared_mem = GetMaxNoOptInSharedMem(device_id);
//
//   std::optional<std::string> error_mess;
//
//   DataBatchHandler my_handler(fft_length, async_batch_size);
//   error_mess = my_handler.PeakAtLastError();
//   if (error_mess) {
//     std::cout << error_mess.value() << std::endl;
//     return std::nullopt;
//   }
//
//   for(int k=0; k<sample_size + warmup_samples; k++){
//     error_mess = my_handler.CopyDataHostToDevice(data.get());
//     if (error_mess) {
//       std::cout << error_mess.value() << std::endl;
//       return std::nullopt;
//     }
//
//     cudaDeviceSynchronize();
//
//     IntervallTimer computation_time;
//     error_mess = ComputeFFT(my_plan, my_handler, max_no_optin_shared_mem);
//     if (error_mess) {
//       std::cout << error_mess.value() << std::endl;
//       return std::nullopt;
//     }
//
//     cudaDeviceSynchronize();
//
//     if (k >= warmup_samples) {
//       runtime.push_back(computation_time.getTimeInNanoseconds());
//     }
//   }
//
//   BenchResult results;
//   results.average_time_ = ComputeAverage(runtime);
//   results.std_deviation_ = ComputeSigma(runtime, results.average_time_);
//
//   RunResults tmp = {mode, base_fft_warps_per_block, r16_warps_per_block,
//                     r2_blocksize, results};
//   return tmp;
// }

// template <typename Integer>
// std::optional<RunResults> Benchmark(const Integer fft_length,
//                                     const int warmup_samples,
//                                     const int sample_size,
//                                     const int async_batch_size,
//                                     const std::string tuner_results_file){
//   std::cout << "Benchmarking fft_length: " << fft_length << std::endl;
//
//   std::vector<float> weights_RE = GetRandomWeights(10, 42);
//   std::vector<float> weights_IM = GetRandomWeights(10, 4242);
//   std::unique_ptr<__half[]> data =
//       CreateSineSuperpostionBatch(fft_length, async_batch_size,
//                                   weights_RE, weights_IM);
//
//   std::vector<double> runtime;
//
//   std::optional<Plan<Integer>> possible_plan =
//       CreatePlan(fft_length, tuner_results_file);
//   Plan<Integer> my_plan;
//   if (possible_plan) {
//     my_plan = possible_plan.value();
//   } else {
//     std::cout << "Plan creation failed" << std::endl;
//     return std::nullopt;
//   }
//
//   int device_id;
//   cudaGetDevice(&device_id);
//   if (!PlanWorksOnDevice(my_plan, device_id)) {
//     std::cout << "Plan doesnt work on device -> configuration skiped."
//     return std::nullopt;
//   }
//
//   int max_no_optin_shared_mem = GetMaxNoOptInSharedMem(device_id);
//
//   std::optional<std::string> error_mess;
//
//   DataBatchHandler my_handler(fft_length, async_batch_size);
//   error_mess = my_handler.PeakAtLastError();
//   if (error_mess) {
//     std::cout << error_mess.value() << std::endl;
//     return std::nullopt;
//   }
//
//   for(int k=0; k<sample_size + warmup_samples; k++){
//     error_mess = my_handler.CopyDataHostToDevice(data.get());
//     if (error_mess) {
//       std::cout << error_mess.value() << std::endl;
//       return std::nullopt;
//     }
//
//     cudaDeviceSynchronize();
//
//     IntervallTimer computation_time;
//     error_mess = ComputeFFT(my_plan, my_handler, max_no_optin_shared_mem);
//     if (error_mess) {
//       std::cout << error_mess.value() << std::endl;
//       return std::nullopt;
//     }
//
//     cudaDeviceSynchronize();
//
//     if (k >= warmup_samples) {
//       runtime.push_back(computation_time.getTimeInNanoseconds());
//     }
//   }
//
//   BenchResult results;
//   results.average_time_ = ComputeAverage(runtime);
//   results.std_deviation_ = ComputeSigma(runtime, results.average_time_);
//
//   RunResults tmp = {my_plan.base_fft_mode_, my_plan.base_fft_warps_per_block_,
//                     my_plan.r16_warps_per_block_, my_plan.r2_blocksize_,
//                     results};
//   return tmp;
// }

//Bench for single fft of cufft
std::optional<BenchResult> BenchmarkCuFFTHalf(long long fft_length,
                                              const int warmup_samples,
                                              const int sample_size){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights_RE = GetRandomWeights(10, 42);
  std::vector<float> weights_IM = GetRandomWeights(10, 4242);
  std::unique_ptr<__half2[]> data =
      CreateSineSuperpostionH2GPU(fft_length, weights_RE, weights_IM,
                                 static_cast<long long>(10));

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
    return std::nullopt;
  }

  size_t size = 0;
  r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1,
                          CUDA_C_16F, nullptr, 1, 1, CUDA_C_16F, 1, &size,
                          CUDA_C_16F);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return std::nullopt;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(__half2),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    IntervallTimer computation_time;

    r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan execution failed." << std::endl;
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  if (cudaPeekAtLastError() != cudaSuccess){
    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    return std::nullopt;
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

//Bench for single fft of cufft
std::optional<BenchResult> BenchmarkCuFFTFloat(long long fft_length,
                                               const int warmup_samples,
                                               const int sample_size){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights_RE = GetRandomWeights(10, 42);
  std::vector<float> weights_IM = GetRandomWeights(10, 4242);
  std::unique_ptr<cufftComplex[]> data =
      CreateSineSuperpostionF2GPU(fft_length, weights_RE, weights_IM,
                                  static_cast<long long>(10));

  std::vector<double> runtime;

  cufftComplex* dptr_data;
  cufftComplex* dptr_results;
  cudaMalloc(&dptr_data, sizeof(cufftComplex) * fft_length);
  cudaMalloc(&dptr_results, sizeof(cufftComplex) * fft_length);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return std::nullopt;
  }

  r = cufftPlan1d(&plan, fft_length, CUFFT_C2C, 1);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed.\n";
    return std::nullopt;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(cufftComplex),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    IntervallTimer computation_time;

    r = cufftExecC2C(plan, dptr_data, dptr_results, CUFFT_FORWARD);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan execution failed.\n";
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  if (cudaPeekAtLastError() != cudaSuccess){
    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    return std::nullopt;
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

//Bench for single fft of cufft
std::optional<BenchResult> BenchmarkCuFFTDouble(int fft_length,
                                                const int warmup_samples,
                                                const int sample_size){
  std::cout << "Benchmarking fft_length: " << fft_length << std::endl;

  std::vector<float> weights_RE = GetRandomWeights(10, 42);
  std::vector<float> weights_IM = GetRandomWeights(10, 4242);
  std::unique_ptr<cufftDoubleComplex[]> data =
      CreateSineSuperpostionD2GPU(fft_length, weights_RE, weights_IM, 10);

  std::vector<double> runtime;

  cufftDoubleComplex* dptr_data;
  cufftDoubleComplex* dptr_results;
  cudaMalloc(&dptr_data, sizeof(cufftDoubleComplex) * fft_length);
  cudaMalloc(&dptr_results, sizeof(cufftDoubleComplex) * fft_length);

  cufftHandle plan;
  cufftResult r;

  r = cufftCreate(&plan);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed." << std::endl;
    return std::nullopt;
  }

  r = cufftPlan1d(&plan, fft_length, CUFFT_Z2Z, 1);
  if (r != CUFFT_SUCCESS) {
    std::cout << "Error! Plan creation failed.\n";
    return std::nullopt;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    IntervallTimer computation_time;

    r = cufftExecZ2Z(plan, dptr_data, dptr_results, CUFFT_FORWARD);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan execution failed.\n";
      return std::nullopt;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  cufftDestroy(plan);
  cudaFree(dptr_results);
  cudaFree(dptr_data);

  if (cudaPeekAtLastError() != cudaSuccess){
    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    return std::nullopt;
  }

  BenchResult results;
  results.average_time_ = ComputeAverage(runtime);
  results.std_deviation_ = ComputeSigma(runtime, results.average_time_);

  return results;
}

// std::optional<BenchResult> BenchmarkCuFFT(long long fft_length,
//                                           const int warmup_samples,
//                                           const int sample_size,
//                                           const int async_batch_size){
//   std::cout << "Benchmarking fft_length: " << fft_length << std::endl;
//
//   std::vector<float> weights_RE = GetRandomWeights(10, 42);
//   std::unique_ptr<__half[]> data =
//       CreateSineSuperpostionH2Batch(fft_length, async_batch_size, weights_RE);
//
//   std::vector<double> runtime;
//
//   __half2* dptr_data;
//   __half2* dptr_results;
//   cudaMalloc(&dptr_data, sizeof(__half2) * fft_length * async_batch_size);
//   cudaMalloc(&dptr_results, sizeof(__half2) * fft_length * async_batch_size);
//
//   cufftHandle plan;
//   cufftResult r;
//
//   r = cufftCreate(&plan);
//   if (r != CUFFT_SUCCESS) {
//     std::cout << "Error! Plan creation failed." << std::endl;
//     return std::nullopt;
//   }
//
//   size_t size = 0;
//   r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1,
//                           CUDA_C_16F, nullptr, 1, 1, CUDA_C_16F,
//                           async_batch_size, &size, CUDA_C_16F);
//   if (r != CUFFT_SUCCESS) {
//     std::cout << "Error! Plan creation failed." << std::endl;
//     return std::nullopt;
//   }
//
//   for(int k=0; k<sample_size + warmup_samples; k++){
//     cudaMemcpy(dptr_data, data.get(),
//                fft_length * sizeof(__half2) * async_batch_size,
//                cudaMemcpyHostToDevice);
//
//     cudaDeviceSynchronize();
//
//     IntervallTimer computation_time;
//
//     r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
//     if (r != CUFFT_SUCCESS) {
//       std::cout << "Error! Plan execution failed." << std::endl;
//       return std::nullopt;
//     }
//
//     cudaDeviceSynchronize();
//
//     if (k >= warmup_samples) {
//       runtime.push_back(computation_time.getTimeInNanoseconds());
//     }
//   }
//
//   cufftDestroy(plan);
//   cudaFree(dptr_results);
//   cudaFree(dptr_data);
//
//   if (cudaPeekAtLastError() != cudaSuccess){
//     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
//     return std::nullopt;
//   }
//
//   BenchResult results;
//   results.average_time_ = ComputeAverage(runtime);
//   results.std_deviation_ = ComputeSigma(runtime, results.average_time_);
//
//   return results;
// }
