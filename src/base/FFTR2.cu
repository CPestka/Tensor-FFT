//Contains the kernel that performs the radix2 steps
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//This kernel performs the radix 2 combination steps if neccessary. Since it
//can not utilize tensor cores it is much slower than the radix 16 kernel and
//is only used to allow the compatibility with all input sizes that are powers
//of 2.
//Each thread computes two complex points of the resulting FFT and thus the
//total number of threads lauched has to equal sub_fft_length i.e. N/2.
//This kernel performs one combination of 2 N/2 sized ffts and thus if there are
//multiple of those needed for one radix step, multiple kernels have to be
//launched and the ptrs to the in/out data have to point to the beginnning of
//the fft that is to be proccessed and not to the global start of the data.
template <typename Integer>
__global__ void Radix2Kernel(__half2* input_data,
                             __half2* output_data,
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
  __half2 point2 = input_data[memory_point2_offset];

  //Multiply point 2 with twiddle factor
  __half modified_point2_RE =
      __hsub(__hmul(point2.x, twiddle_RE), __hmul(point2.y, twiddle_IM));
  __half modified_point2_IM =
      __hfma(point2.x , twiddle_IM, __hmul(point2.y, twiddle_RE));

  //Load point 1 from global mem once to use it twice
  __half2 point1 = input_data[memory_point1_offset];

  //Combine FFTs (sequential scaling is applied as well) and save results
  __half2 result;
  result.x =
      __hmul(__hadd(point1.x, modified_point2_RE), static_cast<__half>(0.5));
  result.y =
      __hmul(__hadd(point1.y, modified_point2_IM), static_cast<__half>(0.5));
  output_data[memory_point1_offset] = result;

  result.x =
      __hmul(__hsub(point1.x, modified_point2_RE), static_cast<__half>(0.5));
  result.y =
      __hmul(__hsub(point1.y, modified_point2_IM), static_cast<__half>(0.5));
  output_data[memory_point2_offset] = result;
}
