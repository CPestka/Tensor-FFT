#pragma once
#include <iostream>
#include <string>
#include <optional>

//Holds the neccesary info for the computation of a fft of a give length.
//Use CreatePlan() to obtain a plan.
struct Plan{
  int fft_length_;
  int amount_of_r16_steps_;
  int amount_of_r2_steps_;
  int transposer_blocksize_;
  //Should probably be = (amount of tenosr cores per SM) if input size allows it
  int dft_warps_per_block_;
  int r16_warps_per_block_;

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
//size and the used GPU. Current decent guesses for good valuse for the
//blocksizes are 256 or 512 and for the warp amounts 4 (if size >= 16^3*2^4;
//else use 1)
//The used algorithm starts allways with a dft step (size 16) followed by one or
//more radix 16 steps and finishes with 0-3 radix 2 steps (to achieve
//compatibility with all inputsizes that are powers of 2 that are large enough)
std::optional<Plan> CreatePlan(int fft_length, int transposer_blocksize,
                               int dft_warps_per_block, int r16_warps_per_block,
                               int r2_blocksize){
  Plan my_plan;

  my_plan.fft_length_ = fft_length;
  my_plan.transposer_blocksize_ = transposer_blocksize;
  if ((fft_length % (dft_warps_per_block * 16 * 16 * 16)) != 0) {
    std::cout << "Error! fft_length must be devisable by dft_warps_per_block * "
              << "16³"  << std::endl;
    return std::nullopt;
  }
  my_plan.dft_warps_per_block_ = dft_warps_per_block;
  if ((fft_length % (r16_warps_per_block * 16 * 16 * 16)) != 0) {
    std::cout << "Error! fft_length must be devisable by r16_warps_per_block * "
              << "16³"  << std::endl;
    return std::nullopt;
  }
  my_plan.r16_warps_per_block_ = r16_warps_per_block;

  if (!IsPowerOf2(fft_length)) {
    std::cout << "Error! Input size has to be a power of 2!" << std::endl;
    return std::nullopt;
  }

  int log2_of_fft_lenght = ExactLog2(fft_length);

  if ((log2_of_fft_lenght / 4) < 3) {
    std::cout << "Error! Input size has to be larger than 4096 i.e. 16^3"
              << std::endl;
    return std::nullopt;
  }

  my_plan.amount_of_r16_steps_ = (log2_of_fft_lenght / 4) - 1;
  my_plan.amount_of_r2_steps_ = log2_of_fft_lenght % 4;

  int tmp = 1;
  for(int i=0; i<my_plan.amount_of_r2_steps_; i++){
    tmp = tmp *2;
  }
  int minimal_r2_blocksize = fft_length / tmp;
  if ((r2_blocksize >= minimal_r2_blocksize) ||
      ((fft_length % minimal_r2_blocksize) != 0)) {
    std::cout << "Error! fft_length has to be devisable by r2_blocksize and has"
              << " r2_blocksize must be >= minimal_r2_blocksize (fft_length / "
              << "2^amount_of_r2_steps_"
              << std::endl;
    return std::nullopt;
  }
  my_plan.r2_blocksize_ = r2_blocksize;
  
  return std::move(my_plan);
}
