#include <iostream>
#include <cstdint>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../base/Plan.cpp"
#include "ComputeFFT.cu"
#include "TestingDataCreation.cu"
#include "../base/Timer.h"

int main(){
  int transpose_blocksize = 256;
  int dft_warps_per_block;
  int r16_warps_per_block;
  int r2_blocksize = 256;

  int sample_size = 100;
  int log2_upper_limit = 31; //30 approx 10^9
  std::vector<std::vector<int64_t>> run_times;

  int fft_length = 16*16*16; //i.e. 2^12
  for(int j=12; j<=log2_upper_limit; j++){
    std::vector<int64_t> tmp;
    for(int i=0; i<sample_size; i++){
      Plan my_plan;
      if (j<15) {
        dft_warps_per_block = 1;
        r16_warps_per_block = 1;
      } else {
        dft_warps_per_block = 4;
        r16_warps_per_block = 4;
      }
      auto tmp = CreatePlan(fft_length, transpose_blocksize,
                            dft_warps_per_block, r16_warps_per_block,
                            r2_blocksize);
      if (tmp.has_value) {
        my_plan = tmp.value;
      } else {
        return false;
      }

      std::vector<float> weights;
      weights.push_back(2.0);
      weights.push_back(1.4);
      std::unique_ptr<__half[]> data = CreateSineSuperpostion(fft_length, weights);

      IntervallTimer timer;
      auto error_mess = ComputeFFT(my_plan, data.get());
      tmp.push_back(timer.getTimeInMicroseconds());
    }
    run_times.push_back(tmp);
    fft_length = fft_length * 2;
  }

  std::vector<double> average;
  std::vector<double> std_dev;

  for(int i=0; i<run_times.size(); i++){
    double tmp = 0;
    for(int j=0; j<run_times[i].size(); j++){
      tmp += (run_times[i][j] / 1000000);
    }
    tmp = tmp / run_times[i].size();
    average.push_back(tmp);
  }

  for(int i=0; i<run_times.size(); i++){
    double tmp = 0;
    for(int j=0; j<run_times[i].size(); j++){
      double tmp1 = (run_times[i][j] / 1000000) - average[i];
      tmp += (tmp1 * tmp1);
    }
    tmp = tmp / (run_times[i].size() - 1);
    std_dev.push_back(tmp);
  }

  WriteBenchResultsToFile(average, std_dev, std::to_string(sample_size));
}
