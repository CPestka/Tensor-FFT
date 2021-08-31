//Contains the kernel that performs the radix2 steps
#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//This kernel performs the radix 2 combination steps if neccessary. Since it
//can not utilize tensor cores it is much slower than the radix 16 kernel and
//is only used to allow the compatibility with all input sizes that are powers
//of 2.
//Each thread computes two complex points of the resulting FFT and thus the
//toatl number of threads lauched has to equal sub_fft_length i.e. N/2.
//This kernel performs one combination of 2 N/2 sized ffts and thus if there are
//multiple of those needed for one radix step, multiple kernels have to be
//launched and the ptrs to the in/out data have to point to the beginnning of
//the fft that is to be proccessed and not to the global start of the data.
template <typename Integer>
__global__ void Radix2Kernel(__half* input_data_RE, __half* input_data_IM,
                             __half* output_data_RE, __half* output_data_IM,
                             Integer sub_fft_length) {
  Integer memory_point1_offset = blockDim.x * blockIdx.x + threadIdx.x;
  Integer memory_point2_offset = memory_point1_offset + sub_fft_length;

  //The twiddle factor for the first point is 1 -> only the second point has to
  //be modified
  //Compute phase = 2PI*i/fft_length = pi * (i/sub_fft_length)
  //Use float to prevent overflow of large ints memory_point1_offset and
  //sub_fft_length
  float tmp = static_cast<float>(memory_point1_offset) /
              static_cast<float>(sub_fft_length);
  //__half phase = __hmul(static_cast<__half>(M_PI), static_cast<__half>(tmp));
  float phase = M_PI * tmp;

  __half twiddle_RE = cosf(phase);
  __half twiddle_IM = -sinf(phase);

  //Fetch current data once from global memory to use it twice
  __half point2_RE = input_data_RE[memory_point2_offset];
  __half point2_IM = input_data_IM[memory_point2_offset];

  //Multiply point 2 with twiddle factor
  __half modified_point2_RE =
      __hsub(__hmul(point2_RE, twiddle_RE), __hmul(point2_IM, twiddle_IM));
  __half modified_point2_IM =
      __hfma(point2_RE , twiddle_IM, __hmul(point2_IM, twiddle_RE));

  //Load point 1 from global mem once to use it twice
  __half point1_RE = input_data_RE[memory_point1_offset];
  __half point1_IM = input_data_IM[memory_point1_offset];

  //Combine FFTs

  //For unscaled or scaling at once
  // output_data_RE[memory_point1_offset] =
  //     __hadd(point1_RE, modified_point2_RE);
  // output_data_IM[memory_point1_offset] =
  //     __hadd(point1_IM, modified_point2_IM);
  //
  // output_data_RE[memory_point2_offset] =
  //     __hadd(point1_RE, modified_point2_RE);
  // output_data_IM[memory_point2_offset] =
  //     __hadd(point1_IM, modified_point2_IM);

  //For sequential scaling
  output_data_RE[memory_point1_offset] =
      __hmul(__hadd(point1_RE, modified_point2_RE), static_cast<__half>(0.5));
  output_data_IM[memory_point1_offset] =
      __hmul(__hadd(point1_IM, modified_point2_IM), static_cast<__half>(0.5));

  output_data_RE[memory_point2_offset] =
      __hmul(__hsub(point1_RE, modified_point2_RE), static_cast<__half>(0.5));
  output_data_IM[memory_point2_offset] =
      __hmul(__hsub(point1_IM, modified_point2_IM), static_cast<__half>(0.5));
}
