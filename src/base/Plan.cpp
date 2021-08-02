#pragma once
#include <iostream>
#include <string>
#include <optional>
#include <fstream>
#include <sstream>

//Holds the neccesary info for the computation of a fft of a given length.
//Use CreatePlan() to obtain a plan.
struct Plan{
  int fft_length_;
  int amount_of_r16_steps_;
  int amount_of_r2_steps_;
  int max_amount_of_warps_;

  int transposer_blocksize_;
  int transposer_amount_of_blocks_;

  int dft_warps_per_block_;
  int dft_amount_of_blocks_;

  int r16_warps_per_block_;
  int r16_amount_of_blocks_;

  int r2_blocksize_;
};

bool IsPowerOf2(int x) {
  if (x==0){
    return false;
  }
  return ((x & (x - 1)) == 0);
}

//Requires x to be power of 2
int ExactLog2(int x) {
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

//Prefered way to create a Plan for a fft of size fft_length. The other
//parameters are performance parameters whos optimal value depends on the fft
//size and the used GPU.
//This overload of the function uses best guesses for the performance parameters
//(i.e. the blocksizes and resulting amount of blocks)
//The used algorithm starts allways with a dft step (size 16) followed by one or
//more radix 16 steps and finishes with 0-3 radix 2 steps (to achieve
//compatibility with all inputsizes that are powers of 2 that are large enough)
//Due to this the input size has to be a power of 2 and at least of size 16^2.
std::optional<Plan> CreatePlan(int fft_length){
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
  my_plan.max_amount_of_warps_ = fft_length / 256;

  my_plan.transposer_blocksize_ = 256;
  my_plan.transposer_amount_of_blocks_ = fft_length / 256;

  my_plan.dft_warps_per_block_ = my_plan.max_amount_of_warps_ <= 4 ?
                                 my_plan.max_amount_of_warps_ : 4;
  my_plan.dft_amount_of_blocks_ = my_plan.max_amount_of_warps_ /
                                  my_plan.dft_warps_per_block_;

  my_plan.r16_warps_per_block_ = my_plan.max_amount_of_warps_ <= 4 ?
                                my_plan.max_amount_of_warps_ : 4;
  my_plan.r16_amount_of_blocks_ = my_plan.max_amount_of_warps_ /
                                  my_plan.r16_warps_per_block_;

  my_plan.r2_blocksize_ = 256;

  return std::move(my_plan);
}

//This overload takes the performance parameters as input
std::optional<Plan> CreatePlan(int fft_length, int transposer_blocksize,
                               int dft_kernel_warps_per_block,
                               int r16_kernel_warps_per_block,
                               int r2_blocksize){
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
  my_plan.max_amount_of_warps_ = fft_length / 256;

  if ((fft_length % transposer_blocksize) != 0) {
    std::cout << "Error! fft_length has to be evenly devisable by "
              << "transposer_blocksize."
              << std::endl;
    return std::nullopt;
  }
  my_plan.transposer_blocksize_ = transposer_blocksize;
  my_plan.transposer_amount_of_blocks_ = fft_length / transposer_blocksize;

  if ((my_plan.max_amount_of_warps_ % dft_kernel_warps_per_block) != 0) {
    std::cout << "Error! max_amount_of_warps_ i.e. fft_length/256 has to be "
              << "evenly devisable by dft_kernel_warps_per_block."
              << std::endl;
    return std::nullopt;
  }
  my_plan.dft_warps_per_block_ = dft_kernel_warps_per_block;
  my_plan.dft_amount_of_blocks_ = my_plan.max_amount_of_warps_ /
                                  my_plan.dft_warps_per_block_;

  if ((my_plan.max_amount_of_warps_ % r16_kernel_warps_per_block) != 0) {
    std::cout << "Error! max_amount_of_warps_ i.e. fft_length/256 has to be "
              << "evenly devisable by r16_kernel_warps_per_block."
              << std::endl;
    return std::nullopt;
  }
  my_plan.r16_warps_per_block_ = r16_kernel_warps_per_block;
  my_plan.r16_amount_of_blocks_ = my_plan.max_amount_of_warps_ /
                                  my_plan.r16_warps_per_block_;

  int tmp = 1;
  for(int i=0; i<my_plan.amount_of_r2_steps_; i++){
    tmp = tmp * 2;
  }
  int smallest_r2_subfft_length = fft_length / tmp;

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

//TODO: overload that reads optimized parameters from file
std::optional<Plan> CreatePlan(int fft_length, std::string tuner_results_file){
  int transpose_blocksize;
  int dft_warp_amount;
  int r16_warp_amount;
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

      if (std::stoi(tmp) == fft_length) {
        found_correct_fft_length = true;

        tmp = "";
        ss >> tmp;
        transpose_blocksize = std::stoi(tmp);

        tmp = "";
        ss >> tmp;
        dft_warp_amount = std::stoi(tmp);

        tmp = "";
        ss >> tmp;
        r16_warp_amount = std::stoi(tmp);

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

  return std::move(CreatePlan(fft_length, transpose_blocksize, dft_warp_amount,
                              r16_warp_amount, r2_blocksize));
}
