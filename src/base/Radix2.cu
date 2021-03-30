//Contains the kernel that performs the radix2 steps
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

//
__global__ void Radix2Kernel(__half* input_data_RE, __half* input_data_IM,
                             __half* output_data_RE, __half* output_data_IM,
                             int kernel_amount, int current_kernel_id,
                             int fft_length, int current_radix2_step,
                             int radix2_loop_length) {

}
