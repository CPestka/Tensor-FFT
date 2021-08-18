//Utility functions and structs for Bench.h Tuner.cu etc.
#pragma once

#include <vector>
#include <optional>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "Bench.h"

struct BenchResult {
  double average_time_;
  double std_deviation_;
};

struct RunConfig {
  BaseFFTMode mode_;
  int base_fft_warps_per_block_;
  int r16_warps_per_block_;
  int r2_blocksize_;
};

struct RunResults {
  BaseFFTMode mode_;
  int base_fft_warps_per_block_;
  int r16_warps_per_block_;
  int r2_blocksize_;
  BenchResult results_;
};

struct RunParameterSearchSpace {
  std::vector<BaseFFTMode> mode_;
  std::vector<int> base_fft_warps_per_block_;
  std::vector<int> r16_warps_per_block_;
  std::vector<int> r2_blocksize_;
};

template <typename T>
T ComputeAverage(const std::vector<T> data){
  T tmp = 0;
  for(int i=0; i<static_cast<int>(data.size()); i++){
    tmp += data[i];
  }
  return (tmp / (static_cast<double>(data.size()) - 1));
}

template <typename T>
T ComputeSigma(const std::vector<T> data, const T average){
  T tmp = 0;
  for(int i=0; i<static_cast<int>(data.size()); i++){
    T tmp_1 = data[i] - average;
    tmp += (tmp_1 * tmp_1);
  }
  return sqrt(tmp / (static_cast<double>(data.size()) - 1));
}

template <typename T>
RunConfig GetFastestConfig(const std::vector<RunResults> &results){
  T smallest = results[0].results_.average_time_;
  int id_smallest = 0;
  for(int i=1; i<static_cast<int>(results.size()); i++){
    if (results[i].results_.average_time_ < smallest) {
      id_smallest = i;
      smallest = results[i].results_.average_time_;
    }
  }
  RunConfig fastest_config = results[id_smallest];
  return fastest_config;
}

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
RunParameterSearchSpace GetSearchSpace(const Integer fft_length, int device_id){
  RunParameterSearchSpace search_space;

  search_space.mode_.push_back(Mode_256);
  if (fft_length >= 4096) {
    search_space.mode_.push_back(Mode_4096);
  }

  cudaDeviceProp properties;
  cudaGetDevicePropertier(&properties, device_id);
  int max_warps_per_block = properties.maxThreadsPerBlock / 32;
  int total_amount_of_warps = fft_length / 32;
  int warp_amount = 1;

  while ((warp_amount <= total_amount_of_warps) &&
         (warp_amount <= max_warps_per_block)){
    search_space.base_fft_warps_per_block_.push_back(warp_amount);
    search_space.r16_warps_per_block_.push_back(warp_amount);
    search_space.r2_blocksize_.push_back(warp_amount * 32);

    warp_amount = (warp_amount * 2);
  }

  return std::move(search_space);
}

//If 1024<maxThreadsPerBlock<2048 -> 256 configs
std::vector<RunConfig> GetRunConfigs(RunParameterSearchSpace search_space){
  std::vector<RunConfig> configs;

  //For the 256 mode
  for(int i=0;
      i<static_cast<int>(search_space.base_fft_warps_per_block_.size()); i++){
    for(int j=0; j<static_cast<int>(search_space.r16_warps_per_block_.size());
        j++){
      for(int k=0; k<static_cast<int>(search_space.r2_blocksize_.size()); k++){
        RunConfig tmp;
        tmp.mode_ = Mode_256;
        tmp.base_fft_warps_per_block_ =
            search_space.base_fft_warps_per_block_[i];
        tmp.r16_warps_per_block_ = search_space.r16_warps_per_block_[j];
        tmp.r2_blocksize_ = search_space.r2_blocksize_[k];
        configs.push_back(tmp);
      }
    }
  }

  //This mode is only avaiable for fft_length >= 4096
  if (search_space.mode_.back() == Mode_4096) {
    //For the 4096 mode the base_fft_warps_per_block_ is always 16
    for(int j=0; j<static_cast<int>(search_space.r16_warps_per_block_.size());
        j++){
      for(int k=0; k<static_cast<int>(search_space.r2_blocksize_.size()); k++){
        RunConfig tmp;
        tmp.mode_ = Mode_4096;
        tmp.base_fft_warps_per_block_ = 16;
        tmp.r16_warps_per_block_ = search_space.r16_warps_per_block_[j];
        tmp.r2_blocksize_ = search_space.r2_blocksize_[k];
        configs.push_back(tmp);
      }
    }
  }

  return std::move(configs);
}

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::vector<RunResults> RunBenchOverSearchSpace(
    const std::vector<RunConfig> &configs,
    const sample_size,
    const warmup_samples,
    const Integer fft_length){
  std::vector<RunResults> results;

  for(int i=0; i<static_cast<int>(configs.size()); i++){
    //Run Bench with a configuration
    std::optional<BenchResult> possible_bench_result = Benchmark(
        fft_length, warmup_samples, sample_size, configs[i].mode_,
        configs[i].base_fft_warps_per_block_,
        configs[i].r16_warps_per_block_, configs[i].r2_blocksize_);

    //If Benchmark completed successfully save results.
    if (possible_bench_result) {
      RunResults current_result =
          {configs[i].mode_, configs[i].base_fft_warps_per_block_,
           configs[i].r16_warps_per_block_, configs[i].r2_blocksize_,
           possible_bench_result.value()};
      results.push_back(current_result);
    }
  }

  return std::move(results);
}

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::vector<RunResults> RunBenchOverSearchSpace(
    const std::vector<RunConfig> &configs,
    const sample_size,
    const warmup_samples,
    const async_batch_size,
    const Integer fft_length){
  std::vector<RunResults> results;

  for(int i=0; i<static_cast<int>(configs.size()); i++){
    //Run Bench with a configuration
    std::optional<BenchResult> possible_bench_result = Benchmark(
        fft_length, warmup_samples, sample_size, async_batch_size,
        configs[i].mode_, configs[i].base_fft_warps_per_block_,
        configs[i].r16_warps_per_block_, configs[i].r2_blocksize_);

    //If Benchmark completed successfully save results.
    if (possible_bench_result) {
      RunResults current_result =
          {configs[i].mode_, configs[i].base_fft_warps_per_block_,
           configs[i].r16_warps_per_block_, configs[i].r2_blocksize_,
           possible_bench_result.value()};
      results.push_back(current_result);
    }
  }

  return std::move(results);
}
