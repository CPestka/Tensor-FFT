//Contains functions that produce example input data for testing of the ffts
#pragma once

#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//Creates half precission data which is a superpostion of the a_i * sin(i * x)
//with x [0:1] and a_i provided by weights
std::unique_ptr<__half[]> CreateSineSuperpostion(int amount_of_timesamples,
                                                 std::vector<float> weights){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(
      2 * amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights.size()); j++){
      tmp += (weights[j] * sin(2 * M_PI * (j + 1) *
             (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i] = __float2half(tmp);
    data[i + amount_of_timesamples] = 0;
  }
  return data;
}
