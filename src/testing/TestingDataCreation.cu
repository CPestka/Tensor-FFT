//Contains functions that produce example input data for testing of the ffts
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <cuda_fp16.h>

//Returns input data for which RE=cos(t) and IM=0.
//Returns in form of two complex fp16 values packed into one __half2
__host__ std::unique_ptr<__half2[]> CreateRealCosineData(
    float amount_of_oscialtions, int amount_of_time_samples){

  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
                                      amount_of_time_samples);

  for(int i=0; i<amount_of_time_samples; i++){
    float phase = 2 * M_PI * amount_of_oscialtions * (i/amount_of_time_samples);
    data[i] = __floats2half2_rn(cos(phase), 0);
  }
  return data;
}

//Returns input data for which RE=cos(t)+cos(2t)+cos(4t)+cos(8t) and IM=0.
//Returns in form of two complex fp16 values packed into one __half2
__host__ std::unique_ptr<__half2[]> Create4RealCosineData(
    float amount_of_oscialtions, int amount_of_time_samples){

  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
                                      amount_of_time_samples);

  for(int i=0; i<amount_of_time_samples; i++){
    float phase = 2 * M_PI * amount_of_oscialtions * (i/amount_of_time_samples);
    data[i] = __floats2half2_rn(cos(phase) + cos(2*phase) + cos(4*phase) +
                                cos(8*phase), 0);
  }
  return data;
}
