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
//This kernel performs one combination of 2 N/2 sized ffts and thus if there are
//multiple of those needed for one radix step, multiple kernels have to be
//launched and the ptrs to the in/out  data have to point to the beginnning of
//the fft that is to be proccessed and not to the global start of the data.
__global__ void Radix2Kernel(__half* input_data_RE, __half* input_data_IM,
                             __half* output_data_RE, __half* output_data_IM,
                             int sub_fft_length) {
  int memory_point1_offset = blockDim.x * blockIdx.x + threadIdx.x;
  int memory_point2_offset = memory_point1_offset + sub_fft_length;

  //The twiddle factor for the first point is 1 -> only the second point has to
  //be modified
  float phase = (-M_PI * memory_point1_offset) / sub_fft_length;
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
