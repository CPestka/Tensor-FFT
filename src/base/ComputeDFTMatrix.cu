#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//This kernel precomputes multiple dft matrices. One of these is used by one
//warp in the kernels DFTKernel() and Radix16Kernel().
//The number of threads has to be equal to the amount of entries to be computed.
//I.e. for 16 of the 16x16 matrices exactly 16*16*16 threads have to be launched
//TO-SELF: If Precomputation is actually faster has to be determined later
__global__ void ComputeDFTMatrix(__half* dft_matrix_batch_RE,
                                 __half* dft_matrix_batch_IM) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int inter_matrix_id = thread_id % (16 * 16);
  int row_id = inter_matrix_id % 16;
  int collum_id = inter_matrix_id / 16;

  float phase = (static_cast<float>(2 * row_id * collum_id) * M_PI) / 16.0;
  dft_matrix_batch_RE[thread_id] = __float2half(cos(phase));
  dft_matrix_batch_IM[thread_id] = __float2half(-sin(phase));
}
