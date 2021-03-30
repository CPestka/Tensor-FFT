//Contains the kernel that performs the radix16 steps on tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

//
__global__ void Radix16Kernel(__half* input_data_RE, __half* input_data_IM,
                          __half* output_data_RE, __half* output_data_IM,
                          int kernel_amount, int current_kernel_id,
                          int fft_length, int current_radix16_step) {

}
