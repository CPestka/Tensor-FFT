//This file contains the kernel TensorFFT that combines the work that can be
//alternatively performed by calling the TransposeKernel followed by the
//DFTKernel and Radix16KernelFirstStep and Radix16Kernel.
//Compared to calling all the mentioned kernels sequentialy it removes the need
//for many global reads and writes by utilizing shared memory as a buffer
//instead, where possible, as well as removing some computations that are not
//neccesary for this design. Also the need for synchronsation between all blocks
//after the kernels TransposeKernel and DFTKernel are no longer neccesary due to
//the usage of shared memory (the still neccesary synchronsations after the R16
//steps is provided via cooperative groups). Further the dft matrices are no
//longer required to be precomputed via the ComputeDFTMatrix kernel which also
//reduces the amount of needed global memory from sizeof(__half)*2*3*fft_length
//to sizeof(__half)*2*2*fft_length bytes.
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
using namespace nvcuda;

__global__ void TensorFFT(__half* input_data_RE, __half* input_data_IM,
                          __half* output_data_RE, __half* output_data_IM,
                          int fft_length, int amount_of_r16_steps){
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  int inter_block_warp_id = warp_id % (blockDim.x / 32);
  //Used to devide work for threads in a warp since the problem size is 16 based
  //and the tensor core operations are "warp wide".
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  extern __shared__ __half buffer[];
  int warp_shared_memory_offset = 1024 * inter_block_warp_id;
  int warp_global_memory_offset = 256 * warp_id;
  __half* buffer_RE = buffer + warp_shared_memory_offset;
  __half* buffer_IM = buffer + warp_shared_memory_offset + 256;
  __half* buffer_tmp_RE = buffer + warp_shared_memory_offset + 512;
  __half* buffer_tmp_IM = buffer + warp_shared_memory_offset + 768;

  //
  //Setup DFT Matrix
  //

  //Declare fragments that will hold the DFT matrix
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;

  //Write DFTMatrix to shared memory
  //TODO: replace computation with literals
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + 8 * inter_warp_id_is_upper_16;
    int buffer_array_id = inter_warp_id_16 + 16 * j;
    __half* phase = __hdiv(__hmul(j * inter_warp_id_16, M_PI), 8.0);
    buffer_RE[buffer_array_id] = hcos(phase);
    buffer_IM[buffer_array_id] = -hsin(phase);
  }

  //Load DFT matrix into the according fragments
  wmma::load_matrix_sync(dft_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, buffer_IM, 16);

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
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
  wmma::fill_fragment(accumulator_IM_frag, 0.0f);

  //Compute x of input_data[x] for a given output_data[thread_id]
  //"Reverse process" of what is done in TransposeKernel(), in the sense that
  //there we compute x of output_data[x] from input_data[thread_id].
  //The reverse is done here to get the needed output_data linear in the
  //thread id so that a wwarp of threads fetches itś own inputs for the next
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
  }

  //Load the inputs
  wmma::load_matrix_sync(dft_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, buffer_IM, 16);

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

  //Load 16 16-point ffts from shared mem buffer, multiply them with according
  //twiddle factors and store them to other shared memory buffer. During that
  //a transpose is also performed.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    int buffer_array_id = (inter_warp_id_16 + 16 * j);
    int buffer_array_id_transposed = (j + 16 * inter_warp_id_16);

    //Compute RE and IM of twiddle factors
    __half phase = __hdiv(__hmul(inter_warp_id_16 * j, M_PI), 128.0);
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
  wmma::load_matrix_sync(data_RE_frag_transposed, buffer_tmp_RE, 16);
  wmma::load_matrix_sync(data_IM_frag_transposed, buffer_tmp_IM, 16);

  //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
  //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
  //RE(A)xIM(B) + IM(A)xRE(B)
  wmma::mma_sync(accumulator_RE_1_frag, dft_RE_frag, data_RE_frag_transposed,
                 accumulator_RE_1_frag);
  wmma::mma_sync(accumulator_RE_2_frag, dft_IM_frag, data_IM_frag_transposed,
                 accumulator_RE_2_frag);
  wmma::mma_sync(accumulator_IM_frag, dft_IM_frag, data_RE_frag_transposed,
                 accumulator_IM_frag);
  wmma::mma_sync(accumulator_IM_frag, dft_RE_frag, data_IM_frag_transposed,
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
    int buffer_array_id = (inter_warp_id_16 + 16 * j);
    int buffer_array_id_transposed = (j + 16 * inter_warp_id_16);

    output_data_RE[buffer_array_id] = buffer_RE[buffer_array_id_transposed];
    output_data_IM[buffer_array_id] = buffer_IM[buffer_array_id_transposed];
  }

  //
  //Perform rest of R16 steps
  //

  int sub_fft_length = 256;

  for(int n=1; n<amount_of_r16_steps; n++){
    int combined_fft_length = sub_fft_length * 16;
    int amount_of_warps_pes_substep = sub_fft_length / 16;
    int inter_substep_id = warp_id % amount_of_warps_pes_substep;
    int substep_id = warp_id / amount_of_warps_pes_substep;

    //SynchBarier for all threads across all blocks
    cooperative_groups::grid_group group = cooperative_groups::this_grid();
    group.sync();

    //Each of the 32 threads pre warp loads 8 (8*32=16*16) data
    //points. However the data of the needed 16x16 matrix of input data is not
    //linaer in memory. The entire 16^mx16 matrix (which is linear in memory) is
    //divided into m 16x16 matrices. This means that the data for one 16x16
    //matrix consists of 16 length 16 linear chuncks, which are offset in
    //respect to each other by sub_fft_length=16^m.
    //Also, by swaping the indecies when loading the storing to and from the
    //fragment the fragment holds the transposed data, which is needed since the
    //data is stored in row major order in memory but is needed in collum major
    //for the matrix multiplication.
    #pragma unroll
    for(int k=0; k<8; k++){
      int i = inter_warp_id_16 + (inter_substep_id * 16);
      int j = k + (8 * inter_warp_id_is_upper_16);
      int global_memory_offset = i +
                                 sub_fft_length * j +
                                 substep_id * combined_fft_length;
      int buffer_matrix_memory_offset = j + 16 * inter_warp_id_16;

      //Compute twiddle factors
      __half phase = __hdiv(__hmul(2 * i * j, M_PI), combined_fft_length);
      //TO-SELF: test __cosf vs cos accuracy and speed
      __half twiddle_RE = hcos(phase);
      __half twiddle_IM = -hsin(phase);

      //Fetch current data once from global memory to use it twice
      __half input_RE = input_data_RE[global_memory_offset];
      __half input_IM = input_data_IM[global_memory_offset];

      //Store modified data to buffer arrays
      //mod_RE = RE*twid_RE - IM*twid_IM
      buffer_RE[buffer_matrix_memory_offset] =
          __hsub(__hmul(input_RE, twiddle_RE), __hmul(input_IM, twiddle_IM));
      //mod_IM = RE*twid_IM + IM*twid_RE
      buffer_IM[buffer_matrix_memory_offset] =
          __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
    }

    //Load the modified data from shared mem buffer
    wmma::load_matrix_sync(data_RE_frag, buffer_RE, 16);
    wmma::load_matrix_sync(data_IM_frag, buffer_IM, 16);

    //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
    //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
    //RE(A)xIM(B) + IM(A)xRE(B)
    wmma::mma_sync(accumulator_RE_1_frag, data_RE_frag, dft_RE_frag,
                   accumulator_RE_1_frag);
    wmma::mma_sync(accumulator_RE_2_frag, data_IM_frag, dft_IM_frag,
                   accumulator_RE_2_frag);
    wmma::mma_sync(accumulator_IM_frag, data_RE_frag, dft_IM_frag,
                   accumulator_IM_frag);
    wmma::mma_sync(accumulator_IM_frag, data_IM_frag, dft_RE_frag,
                   accumulator_IM_frag);

    //Store results to buffer
    wmma::store_matrix_sync(buffer_IM, accumulator_IM_frag, 16,
                            wmma::mem_row_major);
    #pragma unroll
    for(int i=0; i<accumulator_RE_1_frag.num_elements; i++){
      buffer_RE[i] = __hsub(accumulator_RE_1_frag.x[i],
                            accumulator_RE_2_frag.x[i]);
    }

    //Store the results in the appropriately reordered way into the output array
    //The data is stored back the way it was intialy the i.e. a 16^mx16 linear=
    //row-major array and is then reinterpreted as a linear in memory FFT of
    //length 16^(m+1)
    //The transpose operation is also reverted.
    #pragma unroll
    for(int k=0; k<8; k++){
      int i = inter_warp_id_16 + (inter_substep_id * 16);
      int j = k + (8 * inter_warp_id_is_upper_16);
      int global_memory_offset = i +
                                 sub_fft_length * j +
                                 substep_id * combined_fft_length;
      int buffer_matrix_memory_offset = j + 16 * inter_warp_id_16;

      output_data_RE[global_memory_offset] =
          buffer_RE[buffer_matrix_memory_offset];
      output_data_IM[global_memory_offset] =
          buffer_IM[buffer_matrix_memory_offset];
    }

    sub_fft_length = combined_fft_length;
  }
}
