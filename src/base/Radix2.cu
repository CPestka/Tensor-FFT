//Contains the kernel that performs the radix2 steps
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//This kernel performs the radix 2 combinattion steps if neccessary. Since it
//can not utilize tensor cores it is much slowwer than the radix 16 kernel and
//is only used to allow for compatibility with all input sizes that are powers
//of 2.
//Each thread computes two complex points of the resulting FFT.
__global__ void Radix2Kernel(__half* input_data_RE, __half* input_data_IM,
                             __half* output_data_RE, __half* output_data_IM,
                             int kernel_memory_offfset, int sub_fft_length,
                             int amount_of_kernels_per_fft) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int memory_point1_offset = kernel_memory_offfset + thread_id;
  int memory_point2_offset = memory_point1_offset + sub_fft_length;
  int data_point_id;
  if (blockDim.x * gridDim.x < sub_fft_length) {
    data_point_id = thread_id;
  } else {
    data_point_id = thread_id + (sub_fft_length / amount_of_kernels_per_fft);
  }

  //The twiddle factor for the first point is 1 -> only the second point has to
  //be modified
  float phase = (-2 * M_PI * data_point_id) / sub_fft_length;
  __half twiddle_RE = __float2half(cosf(phase));
  __half twiddle_IM = __float2half(sinf(phase));

  //Fetch current data once from global memory to use it twice
  __half input_RE = input_data_RE[memory_point2_offset];
  __half input_IM = input_data_IM[memory_point2_offset];

  //Multiply point 2 with twiddle factor
  __half modified_point2_RE =  input_RE * twiddle_RE - input_IM * twiddle_IM;
  __half modified_point2_IM =  input_RE * twiddle_IM + input_IM * twiddle_RE;

  //Combine FFTs
  output_data_RE[memory_point1_offset] = input_data_RE[memory_point1_offset] +
                                         modified_point2_RE;
  output_data_IM[memory_point1_offset] = input_data_IM[memory_point1_offset] +
                                         modified_point2_IM;

  output_data_RE[memory_point2_offset] = input_data_RE[memory_point1_offset] -
                                         modified_point2_RE;
  output_data_IM[memory_point2_offset] = input_data_IM[memory_point1_offset] -
                                         modified_point2_IM;
}
