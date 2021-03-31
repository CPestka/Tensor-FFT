//Contains the kernel that performs the baselayer DFT on 16 points each using
//tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

//TODO description
__global__ void DFTKernel(__half2* input_data, __half* output_data_RE,
                          __half* output_data_IM, int amount_of_kernels,
                          int current_kernel_id, int fft_length) {
  int memory_offset = (fft_length / amount_of_kernels) * current_kernel_id;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int amount_of_warps = (gridDim.x * blockDim.x) / 32;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;

  //Shared memory array
  extern __shared__ __half2[] shared_mem;

  //When multiple shared memory arrays are needed it has to be done via ptrs
  //the "main" shared memory array
  __half2* input = shared_mem;
  int amount_of_elements_per_kernel = 256 * amount_of_warps;
  __half* input_RE = (__half*)(shared_mem + amount_of_elements_per_kernel);
  __half* input_IM = input_RE + amount_of_elements_per_kernel;
  __half* output_RE = input_IM + amount_of_elements_per_kernel;
  __half* output_IM = output_RE + amount_of_elements_per_kernel;

  //TO-SELF: packing more half2 into smth larger for cpy might be faster
  //Every thread performs 8 cpys -> 8*32=256=16*16 => each warp copies its 16x16
  //matrix
  #pragma unroll
  for(int i=0; i<8; i++){
    int current_id = (warp_id * 256) + (inter_warp_id * 8) + i;
    //Copy packed __half2 data into shared memory
    input[current_id] = input_data[memory_offset + current_id];
    //Unpack data into RE and IM
    input_RE[current_id] = __high2half(input[current_id);
    input_IM[current_id] = __low2half(input[current_id);
  }
  __syncwarp();

  //TODO:Hardcoded dft matrices; make shared so it doesnt exist for all threads
  __half dft_RE[256] = {};
  __half dft_IM[256] = {};

  //Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      data_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      data_IM_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator_RE_1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator_RE_2_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator_IM_frag;

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
  wmma::fill_fragment(accumulator_IM_frag, 0.0f);

  //Load the inputs
  wmma::load_matrix_sync(dft_RE_frag, dft_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, dft_IM, 16);
  int matrix_offset = warp_id * 256;
  wmma::load_matrix_sync(data_RE_frag, input_RE + matrix_offset, 16);
  wmma::load_matrix_sync(data_IM_frag, input_IM + matrix_offset, 16);

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
  wmma::store_matrix_sync(output_data_IM + matrix_offset, accumulator_IM_frag,
                          16, wmma::mem_row_major);

  //RE(A)xRE(B) - IM(A)xIM(B) has to be performed which cant be done directly
  //with the fragments. Instead temporarily save the results in the input
  //arrays, since the data there isnt needed anymore and then perform the
  //subtraction and store the result.
  wmma::store_matrix_sync(input_RE + matrix_offset, accumulator_RE_1_frag,
                          16, wmma::mem_row_major);
  wmma::store_matrix_sync(input_IM + matrix_offset, accumulator_RE_2_frag,
                          16, wmma::mem_row_major);

  #pragma unroll
  for(int i=0; i<8; i++){
   int current_id = (warp_id * 256) + (inter_warp_id * 8) + i;
   output_data_RE[memory_offset + current_id] =
       input_RE[current_id] - input_IM[current_id];
  }
}
