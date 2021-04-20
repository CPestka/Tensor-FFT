//Contains the kernel that performs the baselayer DFT on 16 points each using
//tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

//The FFT algorithm (here implemented via Radix16Kernel() and Radix2Kernel())
//rely on smaller already computed ffts for the computation of the larger fft.
//In this implementation this kernel provides the "first" actualy computed ffts
//by computing 16 point dfts via multiplication with the according dft matrix
//i.e. applying the definition of the dft (an alternative is to go down to ffts
//of size 1, where the dft(x)=x, and doing another recombination step. This isnt
//done here since one DFTKernel() step should be cheaper than a modified
//Radix16Kernel() step)
//Since the tensor cores that are used here can not perform complex
//multiplications directly, the data is instead split into RE and IM part which
//are then used to compute the complex multiplication. Due to this the data is
//from this kernel on is used in form of 2 __half arrays instead of 1 __half2
//array.
//Each warp loads its corresponding 2*16 16x16 matrices of data into the
//fragments, which are used by nvidias wmma api to hold the data for the tensor
//core matrix multiplication. The multiplications are then performed and the
//results are stored back to memory.
//TO-SELF: Not sure which memory solution best for dft_matrix data. 3 possible
//sloutions: 1. 1 fragment in global memory + low memory usage - all warps read
//              same data -> conflict while loading fragments
//           2. 1 fragment for each warp in global memory + no conflict - uses
//              same amount of memory than whole input data - memory requirement
//              on GPU from 2x inputdata size to 3X and additional slow memcpy
//              to device
//           3. Just compute it in kernel
//Currently used is version 1.
__global__ void DFTKernel(__half* input_data_RE, __half* input_data_IM,
                          __half* output_data_RE, __half* output_data_IM,
                          __half* dft_matrix_batch_RE,
                          __half* dft_matrix_batch_IM,
                          int amount_of_kernels, int current_kernel_id,
                          int fft_length) {
  int memory_kernel_offset =
      (fft_length / amount_of_kernels) * current_kernel_id;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;

  //Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      data_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      data_IM_frag;
  //TO-SELF:curretnly acc into half vs float -> worse accuracy vs needing
  //conversion
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_2_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_IM_frag;

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
  wmma::fill_fragment(accumulator_IM_frag, 0.0f);

  //Load the inputs
  wmma::load_matrix_sync(dft_RE_frag, dft_matrix_batch_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, dft_matrix_batch_IM, 16);
  //The data for one Kernel launch beginns at input_data_x + memory_offset
  //Each warp then uses one 16x16x16 matrix batch
  int total_memory_offset = memory_kernel_offset + (warp_id * 4096);
  wmma::load_matrix_sync(data_RE_frag, input_data_RE + total_memory_offset, 16);
  wmma::load_matrix_sync(data_IM_frag, input_data_IM + total_memory_offset, 16);

  //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
  //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
  //RE(A)xIM(B) + IM(A)xRE(B)
  wmma::mma_sync(accumulator_RE_1_frag, dft_RE_frag, data_RE_frag,
                 accumulator_RE_1_frag);
  wmma::mma_sync(accumulator_RE_2_frag, dft_IM_frag, data_IM_frag,
                 accumulator_RE_2_frag);
  wmma::mma_sync(accumulator_IM_frag, dft_RE_frag, data_IM_frag,
                 accumulator_IM_frag);
  wmma::mma_sync(accumulator_IM_frag, dft_IM_frag, data_RE_frag,
                 accumulator_IM_frag);

  //Store IM part of the output
  wmma::store_matrix_sync(output_data_IM + total_memory_offset,
                          accumulator_IM_frag, 16, wmma::mem_row_major);

  //RE(A)xRE(B) - IM(A)xIM(B) has to be performed which cant be done directly
  //with the fragments. Instead temporarily save the results in 2 buffer arrays,
  //compute the differnce and store it.
  extern __shared__ __half shared_memory[];
  __half* buffer1 = shared_memory + warp_id * 8192;
  __half* buffer2 = shared_memory + warp_id * 8192 + 4096;
  //16*16*16*2*4*2 = 64KB max mememory for these buffers per SM (4 TC per SM)
  //A100 164KB register file -> fine on that

  wmma::store_matrix_sync(buffer1, accumulator_RE_1_frag, 16,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(buffer2, accumulator_RE_2_frag, 16,
                          wmma::mem_row_major);

  //Each thread in a warp computes 128 values and stores them -> 32*128=4096=
  //16*16*16
  #pragma unroll
  for(int i=0; i<128; i++){
    int current_id = inter_warp_id * 128 + i;
    output_data_RE[total_memory_offset + current_id] =
        __hsub(buffer1[current_id], buffer2[current_id]);
  }
}
