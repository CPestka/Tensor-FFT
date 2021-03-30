//Contains the kernel that performs the baselayer DFT on 16 points each using
//tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

//
__global__ void DFTKernel(__half2* input_data, __half* output_data_RE,
                          __half* output_data_IM, int kernel_amount,
                          int current_kernel_id, int fft_length) {

}
