//Contains the Plan struct and related functions. A plan is needed for the
//execution of the FFT similar to the API of e.g. cufft or FFTW3
#pragma once

#include <iostream>
#include <string>
#include <optional>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct LaunchConfig {
  int warps_per_block_;
  int blocksize_;
  int gridsize_;
  int shared_mem_in_bytes_;
};

//Holds the neccesary info for the computation of a fft of a given length.
//Use CreatePlan(...) to obtain a plan.
struct Plan{
  int64_t fft_length_;
  int amount_of_r16_steps_;
  int amount_of_r2_steps_;
  int amount_of_r16_kernels_;

  //True if fft result is in result array false when in input array
  bool results_in_results_;

  LaunchConfig transpose_config_;
  LaunchConfig base_fft_config_;
  LaunchConfig r16_config_;

  std::vector<int> sub_fft_length_;

  int r2_blocksize_;
  std::vector<int> amount_of_r2_kernels_per_r2_step_;
};

template <typename Integer>
bool IsPowerOf2(const Integer x) {
  if (x==0){
    return false;
  }
  return ((x & (x - 1)) == 0);
}

//Requires x to be power of 2
template <typename Integer>
int ExactLog2(const Integer x) {
  if (x == 1) {
    return 0;
  }

  int tmp = x;
  int i = 1;

  while (true) {
    if (((tmp/2) % 2) == 0) {
      i++;
      tmp = tmp / 2;
    } else {
      return i;
    }
  }
}

std::optional<Plan> MakePlan(
    int64_t fft_length,
    const int transpose_warps_per_block = 16,
    const int r16_warps_per_block = 16,
    const int r2_blocksize = 1024){
  Plan my_plan;

  if (!IsPowerOf2(fft_length)) {
    std::cout << "Error! Input size has to be a power of 2!" << std::endl;
    return std::nullopt;
  }

  int log2_of_fft_lenght = ExactLog2(fft_length);

  if (log2_of_fft_lenght < 12) {
    std::cout << "Error! Input size has to be larger than 4096 i.e. 16^3"
              << std::endl;
    return std::nullopt;
  }
  my_plan.fft_length_ = fft_length;

  my_plan.amount_of_r16_steps_ = (log2_of_fft_lenght / 4) - 1;
  my_plan.amount_of_r2_steps_ = log2_of_fft_lenght % 4;
  my_plan.amount_of_r16_kernels_ = my_plan.amount_of_r16_steps_ - 2;

  int total_memory_swaps = 1 + //transpose kernel
                           1 + //base layer fftt kernel
                           my_plan.amount_of_r16_kernels_ +
                           my_plan.amount_of_r2_steps_;
  //These kernels each write their results in the other array that is then used
  //as the input for the next kernel: final result position ->
  my_plan.results_in_results_ = (total_memory_swaps % 2) == 1 ? true : false;

  my_plan.transpose_config_ = {16,
                               16 * 32,
                               fft_length / static_cast<int64_t>(4096),
                               32768};
  my_plan.base_fft_config_ = {16,
                              16 * 32,
                              fft_length / static_cast<int64_t>(4096),
                              32768};

  if (!(r16_warps_per_block == 1 ||
        r16_warps_per_block == 2 ||
        r16_warps_per_block == 4 ||
        r16_warps_per_block == 8 ||
        r16_warps_per_block == 16)) {
    std::cout << "Error! r16_warps_per_block_ != {1,2,4,8,16}." << std::endl;
    return std::nullopt;
  }
  if (r16_warps_per_block < (fft_length / 256)) {
    std::cout << "Error more warps per block for r16 kernel than total warps!"
              << std::endl;
    return std::nullopt;
  }
  my_plan.r16_config_ = {r16_warps_per_block,
                         r16_warps_per_block * 32,
                         (fft_length / static_cast<int64_t>(256))
                          / static_cast<int64_t>(r16_warps_per_block),
                         1536 * r16_warps_per_block};

  for(int i=0; i<my_plan.amount_of_r16_kernels_; i++){
    if (i == 0) {
      my_plan.sub_fft_length_.push_back(4096);
    } else {
      my_plan.sub_fft_length_.push_back(my_plan.sub_fft_length_.back() * 16);
    }
  }
  for(int i=0; i<my_plan.amount_of_r2_steps_; i++){
    if ((i == 0) && (my_plan.amount_of_r16_kernels_ == 0)) {
      my_plan.sub_fft_length_.push_back(4096);
    } else {
      my_plan.sub_fft_length_.push_back(my_plan.sub_fft_length_.back() * 2);
    }
  }

  int tmp = 1;
  my_plan.amount_of_r2_kernels_per_r2_step_.push_back(tmp);
  for(int i=0; i<my_plan.amount_of_r2_steps_; i++){
    tmp = tmp * 2;
    my_plan.amount_of_r2_kernels_per_r2_step_.push_back(tmp);
  }

  if (r2_blocksize > 4096) {
    std::cout << "Error. r2_blocksize > 4096." << std::endl;
    return std::nullopt;
  }
  my_plan.r2_blocksize_ = r2_blocksize;

  return my_plan;
}

bool PlanWorksOnDevice(const Plan my_plan, const int device_id){
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device_id);

  if (properties.major < 8) {
    std::cout << "Error! Compute capability >= 8 is required." << std::endl;
    return false;
  }
  if (properties.major >= 9) {
    std::cout << "Warning! Developted for compute capability 8.0 and 8.6."
              << "If tensor core size = 16x16"
              << "It should work(TM) :)" << std::endl;
  }

  if (properties.warpSize != 32) {
    std::cout << "Error! Warp size of 32 required." << std::endl;
    return false;
  }

  if (((my_plan.transpose_config_.blocksize_ > properties.maxThreadsPerBlock) ||
       (my_plan.base_fft_config_.blocksize_ > properties.maxThreadsPerBlock)) ||
      (my_plan.r16_config_.blocksize_ > properties.maxThreadsPerBlock)) {
    std::cout << "Error! One or more kernels exceeds max threads per block."
              << std::endl;
    return false;
  }

  if (((my_plan.transpose_config_.shared_mem_in_bytes_ >
        static_cast<int>(properties.sharedMemPerBlockOptin)) ||
       (my_plan.base_fft_config_.shared_mem_in_bytes_ >
        static_cast<int>(properties.sharedMemPerBlockOptin))) ||
       (my_plan.r16_config_.shared_mem_in_bytes_ >
         static_cast<int>(properties.sharedMemPerBlockOptin))) {
    std::cout << "Error! One or more kernels exceeds max shared memory per"
              << " block."
              << std::endl;
    return false;
  }

  return true;
}

int GetMaxNoOptInSharedMem(const int device_id){
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device_id);

  return properties.sharedMemPerBlock;
}
