//Used to produce different example input data for the FFT
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//Should only be used vec2 float types with the data in members x and y, like
//float2 etc.
//Total amount of threads has to == fft_length
template <typename Integer, typename float2_t>
__global__ void SineSupperposition(Integer fft_length,
                                   float2_t* output,
                                   float2* weights,
                                   int amount_of_weights,
                                   double normalization_factor = 1.0){
  Integer thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  double y_RE = 0;
  double y_IM = 0;
  double x = static_cast<double>(thread_id) / static_cast<double>(fft_length);
  double two_pi = 2 * static_cast<double>(M_PI);

  for(int i=0; i<amount_of_weights; i++){
    double tmp = sin(two_pi * x * i);
    y_RE += (weights[i].x * tmp);
    y_IM += (weights[i].y * tmp);
  }

  y_RE = y_RE / normalization_factor;
  y_IM = y_IM / normalization_factor;

  output[thread_id].x = y_RE;
  output[thread_id].y = y_IM;
}
