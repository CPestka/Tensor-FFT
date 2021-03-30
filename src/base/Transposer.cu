//Contains the kernel that performs all needed transpose operations on the fft
//input data
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

//
__global__ void TransposeKernel(__half2* input_data, __half2* output_data,
                                int kernel_amount, int current_kernel_id,
                                int fft_length) {

}
