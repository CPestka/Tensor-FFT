//Used to test functonality
#include <iostream>
#include <cstdint>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../base/Plan.cpp"
#include "../ComputeFFT.cu"
#include "TestingDataCreation.cu"
#include "../base/Timer.h"
#include "FileWriter.cu"

int main(){
  int fft_length = 16*16*16*16*16*2*2;
  int transpose_blocksize = 256;
  int dft_warps_per_block = 4;
  int r16_warps_per_block = 4;
  int r2_blocksize = 256;
  int amount_of_ffts = 10;

  std::vector<Plan> my_plan;
  for(int i=0; i<amount_of_ffts; i++){
    auto tmp = CreatePlan(fft_length, transpose_blocksize, dft_warps_per_block,
                          r16_warps_per_block, r2_blocksize);
    if (tmp.has_value) {
      my_plan.push_back(tmp.value);
    } else {
      return false;
    }
  }

  std::vector<float> weights;
  weights.push_back(2.0);
  weights.push_back(1.4);
  std::vector<__half*> data;
  for(int i=0; i<amount_of_ffts; i++){
    data.push_back(CreateSineSuperpostion(fft_length, weights).get());
  }

  IntervallTimer timer;
  auto error_mess = ComputeFFTs(my_plan, data);
  std::cout << "Computation took: " << timer.getTimeInMilliseconds() << " ms"
            << std::endl;

  if (error_mess.has_value) {
    std::cout << error_mess.value << std::endl;
  } else {
    for(int i=0; i<amount_of_ffts; i++){
      WriteResultsToFile("results" + std::to_string(i) + ".dat",
                         my_plan[i].fft_length_, data[i]);
    }
  }
}
