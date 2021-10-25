//Used to produce different example input data for the FFT
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//Should only be used for __half2, float2, double2 for float2_t (no C++20 ->
//no concepts)
//Total amount of threads has to == fft_length
template <typename float2_t>
__global__ void SineSupperposition(long long fft_length,
                                   float2_t* output,
                                   float2* weights,
                                   int amount_of_weights,
                                   double normalization_factor = 1.0){
  long long thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  double y_RE = 0;
  double y_IM = 0;
  double x = static_cast<double>(fft_length) / static_cast<double>(thread_id);
  double 2_Pi = 2 * static_cast<double>(M_PI);

  for(int i=0; i<amount_of_weights; i++){
    double tmp = sin(2_Pi * x * (i+1));
    y_RE = weights[i].x * tmp;
    y_IM = weights[i].y * tmp;
  }

  y_RE = y_RE / normalization_factor;
  y_IM = y_IM / normalization_factor;

  output[thread_id].x = y_RE;
  output[thread_id].y = y_IM;
}
