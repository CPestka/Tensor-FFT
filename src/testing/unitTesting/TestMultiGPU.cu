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
  int amount_of_ffts = 10;
  int amount_of_devices = 4;

  std::vector<int> device_id_list;
  for(int i=0; i<amount_of_devices; i++){
    device_id_list.push_back(i);
  }

  std::vector<std::vector<Plan>> my_plans;
  for(int j=0; j<amount_of_devices; j++){
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
    my_plans.push_back(my_plan);
  }

  std::vector<float> weights;
  weights.push_back(2.0);
  weights.push_back(1.4);
  std::vector<std::vector<__half*>> data;
  for(int j=0; j<amount_of_devices; j++){
    std::vector<__half*> tmp;
    for(int i=0; i<amount_of_ffts; i++){
      tmp.push_back(CreateSineSuperpostion(fft_length, weights).get());
    }
    data.push_back(tmp);
  }

  IntervallTimer timer;
  ComputeFFTsMultiGPU(device_id_list, my_plans, data);
  std::cout << "Computation took: " << timer.getTimeInMilliseconds() << " ms"
            << std::endl;

  for(int j=0; j<amount_of_devices; j++){
    for(int i=0; i<amount_of_ffts; i++){
      WriteResultsToFile(
            "device" + std::to_string(j) + "fft" + std::to_string(i) + ".dat",
            my_plan[j][i].fft_length_, data[j][i]);
    }
  }
}
