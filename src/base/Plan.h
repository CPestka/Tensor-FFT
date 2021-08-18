#pragma once

#include <iostream>
#include <string>
#include <optional>
#include <fstream>
#include <sstream>
#include <type_traits>

enum BaseFFTMode {Mode_256, Mode_4096};

//Holds the neccesary info for the computation of a fft of a given length.
//Use CreatePlan(...) to obtain a plan.
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
struct Plan{
  Integer fft_length_;
  int amount_of_r16_steps_;
  int amount_of_r2_steps_;

  BaseFFTMode base_fft_mode_;
  //True if fft result is in result array false when in input array
  bool results_in_results_;

  int base_fft_warps_per_block_;
  int base_fft_blocksize_;
  int base_fft_gridsize_;
  int base_fft_shared_mem_in_bytes_;

  int r16_warps_per_block_;
  int r16_blocksize_;
  int r16_gridsize_;
  int r16_shared_mem_in_bytes_;

  int r2_blocksize_;
};

template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
bool IsPowerOf2(const Integer x) {
  if (x==0){
    return false;
  }
  return ((x & (x - 1)) == 0);
}

//Requires x to be power of 2
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
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

//Prefered way to create a Plan for a fft of size fft_length.
//The used algorithm starts allways with a dft step (size 16) followed by one or
//more radix 16 steps and finishes with 0-3 radix 2 steps (to achieve
//compatibility with all inputsizes that are powers of 2 that are large enough)
//Due to this the input size has to be a power of 2 and at least of size 16^2.
//For better performance than provided by the default prarameters use your own
//parameters or use the overload that reads parameters from file that can be
//generated by running Tuner.cu.
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<Plan> CreatePlan(const Integer fft_length,
                               const BaseFFTMode mode = Mode_256,
                               const int base_fft_warps_per_block = 8,
                               const int r16_warps_per_block = 8,
                               const int r2_blocksize = 256){
  Plan my_plan;

  if (!IsPowerOf2(fft_length)) {
    std::cout << "Error! Input size has to be a power of 2!" << std::endl;
    return std::nullopt;
  }

  int log2_of_fft_lenght = ExactLog2(fft_length);

  if (log2_of_fft_lenght < 8) {
    std::cout << "Error! Input size has to be larger than 256 i.e. 16^2"
              << std::endl;
    return std::nullopt;
  }
  my_plan.fft_length_ = fft_length;

  my_plan.amount_of_r16_steps_ = (log2_of_fft_lenght / 4) - 1;
  my_plan.amount_of_r2_steps_ = log2_of_fft_lenght % 4;

  if ((mode == Mode_4096) && (fft_length_ < 4096)) {
    std::cout << "Error! Baselayer fft length cant be longer that fft_length."
              << std::endl;
    return std::nullopt;
  }
  my_plan.base_fft_mode_ = mode;

  int remaining_radix_steps_after_base_layer =
      (my_plan.base_fft_mode_ == Mode_256) ?
          (my_plan.amount_of_r16_steps_ + my_plan.amount_of_r2_steps_ - 1) :
          (my_plan.amount_of_r16_steps_ + my_plan.amount_of_r2_steps_ - 2);

  my_plan.results_in_results_ =
      ((remaining_radix_steps_after_base_layer % 2) == 0) ? true : false;

  int total_amount_of_warps = fft_length / 256;

  if (total_amount_of_warps < base_fft_warps_per_block) {
    std::cout << "Warning! Input of base_fft_warps_per_block of "
              << base_fft_warps_per_block
              << " has been overwritten to "
              << total_amount_of_warps
              << " since it is larger than total_amount_of_warps."
              << sttd::endl;
    my_plan.base_fft_warps_per_block_ = total_amount_of_warps;
  } else {
    if ((total_amount_of_warps % base_fft_warps_per_block) != 0) {
      std::cout << "Error! Total amount of warps (fft_length/256) has to be "
                << "evenly devisable by base_fft_warps_per_block."
                << std::endl;
      return std::nullopt;
    }

    if (my_plan.base_fft_mode_ == Mode_4096) {
      if (base_fft_warps_per_block != 16) {
        std::cout << "Warning! input of base_fft_warps_per_block has been"
                  << " overwritten to 16. Since 16 is madatory for mode=4096."
      }
      my_plan.base_fft_warps_per_block_ = 16;
    } else {
      my_plan.base_fft_warps_per_block_ = base_fft_warps_per_block;
    }
  }

  my_plan.base_fft_blocksize_ = my_plan.base_fft_warps_per_block_ * 32;
  my_plan.base_fft_gridsize_ =
      total_amount_of_warps / my_plan.base_fft_warps_per_block_;
  my_plan.base_fft_shared_mem_in_bytes_ =
      my_plan.base_fft_warps_per_block_ * 1024 * sizeof(__half);

  if (total_amount_of_warps < r16_warps_per_block) {
    std::cout << "Warning! Input of r16_warps_per_block of "
              << r16_warps_per_block
              << " has been overwritten to "
              << total_amount_of_warps
              << " since it is larger than total_amount_of_warps."
              << sttd::endl;
    my_plan.r16_warps_per_block_ = total_amount_of_warps;
  } else {
    if ((total_amount_of_warps % r16_warps_per_block) != 0) {
      std::cout << "Error! Total amount of warps (fft_length/256) has to be "
                << "evenly devisable by amount_of_r16_warps_per_block."
                << std::endl;
      return std::nullopt;
    }
    my_plan.r16_warps_per_block_ = r16_warps_per_block;
  }

  my_plan.r16_blocksize_ = my_plan.r16_warps_per_block_ * 32;
  my_plan.r16_gridsize_ =
      total_amount_of_warps / my_plan.r16_warps_per_block_;

  my_plan.r16_shared_mem_in_bytes_ =
      my_plan.r16_warps_per_block_ * 512 * sizeof(__half);

  int tmp = 1;
  for(int i=0; i<my_plan.amount_of_r2_steps_; i++){
    tmp = tmp * 2;
  }
  Integer smallest_r2_subfft_length = fft_length / tmp;

  if ((smallest_r2_subfft_length % r2_blocksize) != 0) {
    std::cout << "Error! smallest_r2_subfft_length i.e. "
              << "pow(2,(log2_of_fft_lenght / 4)) has to be "
              << "evenly devisable by r2_blocksize."
              << std::endl;
    return std::nullopt;
  }
  my_plan.r2_blocksize_ = r2_blocksize;

  return std::move(my_plan);
}

//Overload that reads optimized parameters from file
template <typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
std::optional<Plan> CreatePlan(const Integer fft_length,
                               const std::string tuner_results_file){
  BaseFFTMode mode;
  int base_fft_warps_per_block;
  int r16_warps_per_block;
  int r2_blocksize;

  std::string line;
  std::ifstream file(tuner_results_file);

  bool found_correct_fft_length = false;

  if (file.is_open()) {
    while (std::getline(file, line)){
      std::stringstream ss;
      std::string tmp;

      ss << line;
      ss >> tmp;

      if (static_cast<Integer>(std::stod(tmp)) == fft_length) {
        found_correct_fft_length = true;

        tmp = "";
        ss >> tmp;
        mode = std::stoi(tmp) == 256 ? Mode_256 : Mode_4096;

        tmp = "";
        ss >> tmp;
        base_fft_warps_per_block = std::stoi(tmp);

        tmp = "";
        ss >> tmp;
        r16_warps_per_block = std::stoi(tmp);

        tmp = "";
        ss >> tmp;
        r2_blocksize = std::stoi(tmp);
      }

      if (found_correct_fft_length) {
        break;
      }
    }

    if (!found_correct_fft_length) {
      std::cout << "Error! Tuner file didnt contain requested fft length."
                << std::endl;
      return std::nullopt;
    }
  } else {
    std::cout << "Error! Failed to open tuner file." << std::endl;
    return std::nullopt;
  }

  return std::move(CreatePlan(fft_length, mode, base_fft_warps_per_block,
                              r16_warps_per_block, r2_blocksize));
}

bool PlanWorksOnDevice(const Plan my_plan, const int device_id){
  cudaDeviceProp properties;
  cudaGetDevicePropertier(&properties, device_id);

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

  if (((my_plan.base_fft_blocksize_ > properties.maxThreadsPerBlock) ||
       (my_plan.r16_blocksize_ > properties.maxThreadsPerBlock)) ||
      (my_plan.r2_blocksize_ > properties.maxThreadsPerBlock)) {
    std::cout << "Error! One or more kernels exceeds max threads per block."
              << std::endl;
    return false;
  }

  if ((my_plan.base_fft_shared_mem_in_bytes_ >
       properties.sharedMemPerBlockOptin) ||
      (my_plan.r16_shared_mem_in_bytes_ >
       properties.sharedMemPerBlockOptin)) {
    std::cout << "Error! One or more kernels exceeds max shared memory per"
              << " block."
              << std::endl;
    return false;
  }

  return true;
}