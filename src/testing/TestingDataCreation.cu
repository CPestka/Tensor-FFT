//Contains functions that produce example input data for testing of the ffts
#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//Creates half precission data which is a superpostion of the a_i * sin(2*i * x)
//with x [0:1] and a_i provided by weights
//Has 2*amount_of_timesamples elements, with the RE ones in the first half
std::unique_ptr<__half[]> CreateSineSuperpostion(int amount_of_timesamples,
                                                 std::vector<float> weights){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(
      2 * amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights.size()); j++){
      tmp += (weights[j] * sin(2 * M_PI * ((j * 2) + 1) *
             (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i] = __float2half(tmp);
    data[i + amount_of_timesamples] = 0;
  }
  return data;
}

//Returns ptr to zero valued data
std::unique_ptr<__half[]> CreateZeros(int length){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(2 * length);

  for(int i=0; i<length; i++){
    data[i] = 0;
    data[i + length] = 0;
  }
  return data;
}

//Creates half precission data which is a superpostion of the a_i * sin(2*i * x)
//with x [0:1] and a_i provided by weights
//Has amount_of_timesamples half2 elements holding one complex value each
std::unique_ptr<__half2[]> CreateSineSuperpostionHE(int amount_of_timesamples,
                                                    std::vector<float> weights){
  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
      amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights.size()); j++){
      tmp += (weights[j] * sin(2 * M_PI * ((j * 2) + 1) *
             (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i] = __floats2half2_rn(tmp, 0);
  }
  return data;
}
