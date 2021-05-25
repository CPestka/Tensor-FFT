//Used to test functonality
#include <iostream>
#include <cstdint>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../base/Plan.cpp"
#include "../base/ComputeFFT.cu"
#include "TestingDataCreation.cu"
#include "../base/Timer.h"
#include "FileWriter.cu"

int main(){
  int fft_length = 16*16*16*16*16*2*2;
  int transpose_blocksize = 256;
  int dft_warps_per_block = 4;
  int r16_warps_per_block = 4;
  int r2_blocksize = 256;

  Plan my_plan;
  std::optional<Plan> tmp = CreatePlan(fft_length, transpose_blocksize,
                                       dft_warps_per_block,
                                       r16_warps_per_block, r2_blocksize);
  if (tmp) {
    my_plan = tmp.value();
  } else {
    return false;
  }

  std::vector<float> weights;
  weights.push_back(2.0);
  weights.push_back(1.4);
  std::unique_ptr<__half[]> data = CreateSineSuperpostion(fft_length, weights);

  IntervallTimer timer;
  auto error_mess = ComputeFFT(my_plan, data.get());
  std::cout << "Computation took: " << timer.getTimeInMilliseconds() << " ms"
            << std::endl;

  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
  } else {
    WriteResultsToFile("results.dat", fft_length, data.get());
  }
}
