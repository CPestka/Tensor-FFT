//This file contains the kernel TensorFFT that combines the work that can be
//alternatively performed by calling the TransposeKernel followed by the
//DFTKernel and Radix16KernelFirstStep.
//Compared to calling all the mentioned kernels sequentialy it removes the need
//for many global reads and writes by utilizing shared memory as a buffer
//instead, where possible, as well as removing some computations that are not
//neccesary for this design. Also the need for synchronsation between all blocks
//after the kernels TransposeKernel and DFTKernel are no longer neccesary due to
//the usage of shared memory.
//Further the dft matrices are no longer required to be precomputed via the
//ComputeDFTMatrix kernel which also reduces the amount of needed global memory
//from sizeof(__half)*2*3*fft_length to sizeof(__half)*2*2*fft_length bytes.
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void TensorFFT256(__half* input_data_RE, __half* input_data_IM,
                             __half* output_data_RE, __half* output_data_IM,
                             int fft_length, int amount_of_r16_steps,
                             int amount_of_r2_steps){
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  int inter_block_warp_id = warp_id % (blockDim.x / 32);
  //Used to devide work for threads in a warp since the problem size is 16 based
  //and the tensor core operations are "warp wide".
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  int warp_global_memory_offset = 256 * warp_id;

  __half buffer_RE[256];
  __half buffer_IM[256];

  //
  //Setup DFT Matrix
  //

  //Declare fragments that will hold the DFT matrix
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;

  //Write DFTMatrix to shared memory
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + 8 * inter_warp_id_is_upper_16;
    int buffer_array_id = inter_warp_id_16 + 16 * j;
    //Modulo version for higher accuracy
    /*
    __half phase =
        __hdiv(__hmul(static_cast<__half>((j * inter_warp_id_16) % 16),
                      static_cast<__half>(M_PI)),
               static_cast<__half>(8.0));
    */
    __half phase =
        __hdiv(__hmul(static_cast<__half>(j * inter_warp_id_16),
                      static_cast<__half>(M_PI)),
               static_cast<__half>(8.0));
    buffer_RE[buffer_array_id] = hcos(phase);
    buffer_IM[buffer_array_id] = -hsin(phase);
  }

  //Load DFT matrix into the according fragments
  wmma::load_matrix_sync(dft_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, buffer_IM, 16);

  //Literal version of dft matrix. eqivalent to constexpr version of double cos
  //or sin cast to __half with according phase
  //Done this way sincve there is no constexpr version of trig func
  /*
  __half dft_matrix_RE[256] = {
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,

    1, 0.923828125, 0.70703125, 0.382568359375,
    0, -0.382568359375, -0.70703125, -0.923828125,
    -1, -0.923828125, -0.70703125, -0.382568359375,
    -0, 0.382568359375, 0.70703125, 0.923828125,

    1, 0.70703125, 0, -0.70703125,
    -1, -0.70703125, -0, 0.70703125,
    1, 0.70703125, 0, -0.70703125,
    -1, -0.70703125, -0, 0.70703125,

    1, 0.382568359375, -0.70703125, -0.923828125,
    -0, 0.923828125, 0.70703125, -0.382568359375,
    -1, -0.382568359375, 0.70703125, 0.923828125,
    0, -0.923828125, -0.70703125, 0.382568359375,

    1, 0, -1, -0,
    1, 0, -1, -0,
    1, 0, -1, -0,
    1, -0, -1, -0,

    1, -0.382568359375, -0.70703125, 0.923828125,
    0, -0.923828125, 0.70703125, 0.382568359375,
    -1, 0.382568359375, 0.70703125, -0.923828125,
    -0, 0.923828125, -0.70703125, -0.382568359375,

    1, -0.70703125, -0, 0.70703125,
    -1, 0.70703125, 0, -0.70703125,
    1, -0.70703125, -0, 0.70703125,
    -1, 0.70703125, -0, -0.70703125,

    1, -0.923828125, 0.70703125, -0.382568359375,
    -0, 0.382568359375, -0.70703125, 0.923828125,
    -1, 0.923828125, -0.70703125, 0.382568359375,
    -0, -0.382568359375, 0.70703125, -0.923828125,

    1, -1, 1, -1,
    1, -1, 1, -1,
    1, -1, 1, -1,
    1, -1, 1, -1,

    1, -0.923828125, 0.70703125, -0.382568359375,
    0, 0.382568359375, -0.70703125, 0.923828125,
    -1, 0.923828125, -0.70703125, 0.382568359375,
    -0, -0.382568359375, 0.70703125, -0.923828125,

    1, -0.70703125, 0, 0.70703125,
    -1, 0.70703125, -0, -0.70703125,
    1, -0.70703125, -0, 0.70703125,
    -1, 0.70703125, -0, -0.70703125,

    1, -0.382568359375, -0.70703125, 0.923828125,
    -0, -0.923828125, 0.70703125, 0.382568359375,
    -1, 0.382568359375, 0.70703125, -0.923828125,
    0, 0.923828125, -0.70703125, -0.382568359375,

    1, -0, -1, 0,
    1, -0, -1, -0,
    1, -0, -1, 0,
    1, 0, -1, 0,

    1, 0.382568359375, -0.70703125, -0.923828125,
    -0, 0.923828125, 0.70703125, -0.382568359375,
    -1, -0.382568359375, 0.70703125, 0.923828125,
    -0, -0.923828125, -0.70703125, 0.382568359375,

    1, 0.70703125, -0, -0.70703125,
    -1, -0.70703125, -0, 0.70703125,
    1, 0.70703125, -0, -0.70703125,
    -1, -0.70703125, 0, 0.70703125,

    1, 0.923828125, 0.70703125, 0.382568359375,
    -0, -0.382568359375, -0.70703125, -0.923828125,
    -1, -0.923828125, -0.70703125, -0.382568359375,
    0, 0.382568359375, 0.70703125, 0.923828125
  };

  __half dft_matrix_IM[256] = {
    -0, -0, -0, -0,
    -0, -0, -0, -0,
    -0, -0, -0, -0,
    -0, -0, -0, -0,

    -0, -0.382568359375, -0.70703125, -0.923828125,
    -1, -0.923828125, -0.70703125, -0.382568359375,
    -0, 0.382568359375, 0.70703125, 0.923828125,
    1, 0.923828125, 0.70703125, 0.382568359375,

    -0, -0.70703125, -1, -0.70703125,
    -0, 0.70703125, 1, 0.70703125,
    0, -0.70703125, -1, -0.70703125,
    -0, 0.70703125, 1, 0.70703125,

    -0, -0.923828125, -0.70703125, 0.382568359375,
    1, 0.382568359375, -0.70703125, -0.923828125,
    -0, 0.923828125, 0.70703125, -0.382568359375,
    -1, -0.382568359375, 0.70703125, 0.923828125,

    -0, -1, -0, 1,
    0, -1, -0, 1,
    0, -1, -0, 1,
    0, -1, -0, 1,

    -0, -0.923828125, 0.70703125, 0.382568359375,
    -1, 0.382568359375, 0.70703125, -0.923828125,
    -0, 0.923828125, -0.70703125, -0.382568359375,
    1, -0.382568359375, -0.70703125, 0.923828125,

    -0, -0.70703125, 1, -0.70703125,
    -0, 0.70703125, -1, 0.70703125,
    0, -0.70703125, 1, -0.70703125,
    -0, 0.70703125, -1, 0.70703125,

    -0, -0.382568359375, 0.70703125, -0.923828125,
    1, -0.923828125, 0.70703125, -0.382568359375,
    -0, 0.382568359375, -0.70703125, 0.923828125,
    -1, 0.923828125, -0.70703125, 0.382568359375,

    -0, -0, 0, -0,
    0, -0, 0, -0,
    0, -0, 0, -0,
    0, 0, 0, -0,

    -0, 0.382568359375, -0.70703125, 0.923828125,
    -1, 0.923828125, -0.70703125, 0.382568359375,
    -0, -0.382568359375, 0.70703125, -0.923828125,
    1, -0.923828125, 0.70703125, -0.382568359375,

    -0, 0.70703125, -1, 0.70703125,
    -0, -0.70703125, 1, -0.70703125,
    0, 0.70703125, -1, 0.70703125,
    -0, -0.70703125, 1, -0.70703125,

    -0, 0.923828125, -0.70703125, -0.382568359375,
    1, -0.382568359375, -0.70703125, 0.923828125,
    -0, -0.923828125, 0.70703125, 0.382568359375,
    -1, 0.382568359375, 0.70703125, -0.923828125,

    -0, 1, -0, -1,
    0, 1, -0, -1,
    0, 1, -0, -1,
    0, 1, 0, -1,

    -0, 0.923828125, 0.70703125, -0.382568359375,
    -1, -0.382568359375, 0.70703125, 0.923828125,
    0, -0.923828125, -0.70703125, 0.382568359375,
    1, 0.382568359375, -0.70703125, -0.923828125,

    -0, 0.70703125, 1, 0.70703125,
    -0, -0.70703125, -1, -0.70703125,
    0, 0.70703125, 1, 0.70703125,
    0, -0.70703125, -1, -0.70703125,

    -0, 0.382568359375, 0.70703125, 0.923828125,
    1, 0.923828125, 0.70703125, 0.382568359375,
    -0, -0.382568359375, -0.70703125, -0.923828125,
    -1, -0.923828125, -0.70703125, -0.382568359375
  };

  wmma::load_matrix_sync(dft_RE_frag, dft_matrix_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, dft_matrix_IM, 16);
  */

  //
  //Load "shuffeld" input data for this warp
  //

  //Declare the data fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      data_RE_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      data_IM_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_2_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_IM_frag;

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0);
  wmma::fill_fragment(accumulator_IM_frag, 0.0);

  //Compute x of input_data[x] for a given output_data[thread_id]
  //"Reverse process" of what is done in TransposeKernel(), in the sense that
  //there we compute x of output_data[x] from input_data[thread_id].
  //The reverse is done here to get the needed output_data linear in the
  //thread id so that a warp of threads fetches itś own inputs for the next
  //stage and  can store it in sharde memory
  #pragma unroll
  for(int k=0; k<8; k++){
    int buffer_array_id = k + 8 * inter_warp_id_is_upper_16 +
                          16 * inter_warp_id_16;
    int output_data_id = buffer_array_id + warp_global_memory_offset;

    int tmp_id = output_data_id;

    int input_array_id = 16 * (tmp_id % 16);
    tmp_id = tmp_id / 16;
    input_array_id += (tmp_id % 16);

    for(int i=0; i<amount_of_r16_steps-1; i++){
      tmp_id = tmp_id / 16;
      input_array_id = (16 * input_array_id) + (tmp_id % 16);
    }

    for(int i=0; i<amount_of_r2_steps; i++){
      tmp_id = tmp_id / 2;
      input_array_id = (2 * input_array_id) + (tmp_id % 2);
    }

    buffer_RE[buffer_array_id] = input_data_RE[input_array_id];
    buffer_IM[buffer_array_id] = input_data_IM[input_array_id];

    //For sequential scaling
    //buffer_RE[buffer_array_id] = __hdiv(input_data_RE[input_array_id],
    //                                    static_cast<__half>(256.0));
    //buffer_IM[buffer_array_id] = __hdiv(input_data_IM[input_array_id],
    //                                    static_cast<__half>(256.0));

    //For scaling in one step
    //buffer_RE[buffer_array_id] = __hdiv(input_data_RE[input_array_id],
    //                                    static_cast<__half>(fft_length));
    //buffer_IM[buffer_array_id] = __hdiv(input_data_IM[input_array_id],
    //                                    static_cast<__half>(fft_length));
  }

  //Load the inputs
  wmma::load_matrix_sync(data_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(data_IM_frag, buffer_IM, 16);

  //
  //Perform DFT of length 16
  //

  //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
  //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
  //RE(A)xIM(B) + IM(A)xRE(B)
  wmma::mma_sync(accumulator_RE_1_frag, data_RE_frag, dft_RE_frag,
                 accumulator_RE_1_frag);
  wmma::mma_sync(accumulator_RE_2_frag, data_IM_frag, dft_IM_frag,
                 accumulator_RE_2_frag);
  wmma::mma_sync(accumulator_IM_frag, data_IM_frag, dft_RE_frag,
                 accumulator_IM_frag);
  wmma::mma_sync(accumulator_IM_frag, data_RE_frag, dft_IM_frag,
                 accumulator_IM_frag);

  //Store IM part of the output
  wmma::store_matrix_sync(buffer_IM, accumulator_IM_frag, 16,
                          wmma::mem_row_major);

  //Compute RE(A)xRE(B)-IM(A)xIM(B)
  //Special access patern for uniform operation on all elements of fragments
  #pragma unroll
  for(int i=0; i<accumulator_RE_1_frag.num_elements; i++){
    buffer_RE[i] = __hsub(accumulator_RE_1_frag.x[i],
                          accumulator_RE_2_frag.x[i]);
  }

  //
  //Perform first R16 step
  //

  __half buffer_tmp_RE[256];
  __half buffer_tmp_IM[256];
  //Load 16 16-point ffts from shared mem buffer, multiply them with according
  //twiddle factors and store them to other shared memory buffer. During that
  //a transpose is also performed.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    int buffer_array_id = (inter_warp_id_16 + 16 * j);
    int buffer_array_id_transposed = (j + 16 * inter_warp_id_16);

    //Compute RE and IM of twiddle factors
    __half phase =
        __hdiv(__hmul(static_cast<__half>(inter_warp_id_16 * j),
                      static_cast<__half>(M_PI)),
               static_cast<__half>(128.0));
    //TO-SELF: test __cosf vs cos accuracy and speed
    __half twiddle_RE = hcos(phase);
    __half twiddle_IM = -hsin(phase);

    __half input_RE = buffer_RE[buffer_array_id];
    __half input_IM = buffer_IM[buffer_array_id];

    //Store modified data to buffer arrays
    //mod_RE = RE*twid_RE - IM*twid_IM
    buffer_tmp_RE[buffer_array_id_transposed] =
        __hsub(__hmul(input_RE, twiddle_RE), __hmul(input_IM, twiddle_IM));

    //mod_IM = RE*twid_IM + IM*twid_RE
    buffer_tmp_IM[buffer_array_id_transposed] =
        __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
  }

  //Load the modified data from shared mem buffer
  wmma::load_matrix_sync(data_RE_frag, buffer_tmp_RE, 16);
  wmma::load_matrix_sync(data_IM_frag, buffer_tmp_IM, 16);

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0);
  wmma::fill_fragment(accumulator_IM_frag, 0.0);

  //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
  //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
  //RE(A)xIM(B) + IM(A)xRE(B)
  wmma::mma_sync(accumulator_RE_1_frag, data_RE_frag, dft_RE_frag,
                 accumulator_RE_1_frag);
  wmma::mma_sync(accumulator_RE_2_frag, data_IM_frag, dft_IM_frag,
                 accumulator_RE_2_frag);
  wmma::mma_sync(accumulator_IM_frag, data_IM_frag, dft_RE_frag,
                 accumulator_IM_frag);
  wmma::mma_sync(accumulator_IM_frag, data_RE_frag, dft_IM_frag,
                 accumulator_IM_frag);

  //Store results to buffer
  wmma::store_matrix_sync(buffer_IM, accumulator_IM_frag, 16,
                          wmma::mem_row_major);
  #pragma unroll
  for(int i=0; i<accumulator_RE_1_frag.num_elements; i++){
    buffer_RE[i] = __hsub(accumulator_RE_1_frag.x[i],
                          accumulator_RE_2_frag.x[i]);
  }

  //Store results into global memory and revert transpose.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    int buffer_array_id_transposed = (j + 16 * inter_warp_id_16);
    //Global id also reverses the transpose
    int global_array_id = (inter_warp_id_16 + 16 * j) +
                          warp_global_memory_offset;

    output_data_RE[global_array_id] = buffer_RE[buffer_array_id_transposed];
    output_data_IM[global_array_id] = buffer_IM[buffer_array_id_transposed];
  }
}
