#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//This kernel is responsible for shuffeling the input data for the DIT FFT.
//The naive transpose approach of directly maping the initial postion of the
//data to the shuffled postion and copying it accordingly results in terrible
//performace as the output postions are scattered far appart for a linear
//sequence of input elements, or vice versa.
//In this approach each block reads 64 initialy linear sequences of 64 elements
//each, that have the particular stride that 1 element from each sequence form
//1 linear 64 sequence for the output, resulting in a total of 64 length 64
//sequences to write back to gloabl memory. Thus the intial and final global
//memory accesses are 256Byte large (element size is 4Byte), which is optimal
//and the shuffling is performed in the significantly faster shared memory.
//ONLY WORKS FOR 16 warps per block i.e. blocksize=512!!! (should be best for
//performance anyways)
template <typename Integer>
__global__ void Transposer(__half2* input_data, __half2* output_data,
                           Integer fft_length, int amount_of_r16_steps,
                           int amount_of_r2_steps){
  //The thread id within a block maps to the id within a row [0:63] and to the
  //id within the 64 specific rows needed for the shuffel
  int inter_block_row_id = threadIdx.x / 64;
  int inter_row_id = threadIdx.x % 64; //[0,7]

  //stride of elments in input data that are linear in second to last dim of
  //output data
  Integer small_row_stride = fft_length / 256;
  //stride of elments in input data that are linear in last dim of output data
  Integer large_row_stride = fft_length / 16;

  //As within one block 64 rows from the intital memory are taken, such that
  //when one takes one specific element from each one receives a sequence that
  //is linear in the output and the last two dimension of the output array are
  //always of size 16, the linear sequence streatches across all of the last and
  //across 1/4 of the second to last dimension. To simplify the index
  //calculation, 4 sequential blocks are bundled together to fully streatch
  //the second to last dim.
  int second_to_last_dim_quadrant_id = blockIdx.x % 4;

  //Data offset per bundle of 4 blocks is 1 row of 64 elements
  Integer data_offset = (blockIdx.x / 4) * 64;

  //Numbering of the name of the shared mem buffer is in different kernels is
  //due to stupid limitations in cuda :)
  extern __shared__ __half2 shared_buffer1[];
  __half2* input_buffer = shared_buffer1;
  __half2* output_buffer = shared_buffer1 + 4096;

  //The sequence of these rows is: 16 rows with stride (i * fft_length / 16)
  //with acending i [0,15]. 4 of these sequences of 16 rows are stored in
  //sequence here. They have a stride of j * fft_length/256 with j acsending
  //[0,15]. Four of these values for j are realised by one block and 4
  //sequential blocks each then realise the entire range of j.
  #pragma unroll
  for(int j=0; j<2; j++){
    for(int i=0; i<4; i++){
      input_buffer[threadIdx.x + (j * 512) + (i * 1024)] = input_data[
          inter_row_id +
          (large_row_stride * (inter_block_row_id + (8 * j))) +
          (small_row_stride * (i + (4 * second_to_last_dim_quadrant_id))) +
          data_offset];
    }
  }

  //Transpose buffer matrix
  //Note: not sure if this way is good in terms of bank conflicts. Solve that
  //later.
  #pragma unroll
  for(int i=0; i<8; i++){
    output_buffer[threadIdx.x + (i * 512)] =
        input_buffer[inter_block_row_id + (8 * i) + 64 * inter_row_id];
  }

  #pragma unroll
  for(int i=0; i<8; i++){
    Integer current_stride = fft_length;
    Integer current_id = inter_block_row_id + (8 * i) + data_offset;
    Integer current_offset = 0;

    for(int k=0; k<amount_of_r2_steps; k++){
      current_stride = current_stride / 2;
      current_offset += (current_stride * (current_id % 2));
      current_id = current_id / 2;
    }

    for(int k=0; k<amount_of_r16_steps-1; k++){
      current_stride = current_stride / 16;
      current_offset += (current_stride * (current_id % 16));
      current_id = current_id / 16;
    }

    current_offset += (second_to_last_dim_quadrant_id * 64) + inter_row_id;

    output_data[current_offset] = output_buffer[threadIdx.x + (512 * i)];
  }

}

//Old transpose kernel for testing (but with half2 instead of half)
__global__ void TransposeKernel(__half2* input_data,
                                __half2* output_data,
                                int fft_length,
                                int amount_of_r16_steps,
                                int amount_of_r2_steps) {
  //The thread id is the id for the entry of the input array we wish to store to
  //the correct position in the output array
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  int output_id = 0;
  int current_row_length = fft_length;
  int tmp = id;

  for(int i=0; i<amount_of_r2_steps; i++){
    current_row_length = current_row_length / 2;
    output_id += ((tmp % 2) * current_row_length);
    tmp = tmp / 2;
  }

  for(int i=0; i<amount_of_r16_steps; i++){
    current_row_length = current_row_length / 16;
    output_id += ((tmp % 16) * current_row_length);
    tmp = tmp / 16;
  }
  output_id += tmp;

  //Move input data to correct position
  output_data[output_id] = input_data[id];
}

//For the case of fft_length=4096
//blocksize == 512
//gridsize == 1
__global__ void Transposer4k(__half2* input_data, __half2* output_data){
  //Numbering of the name of the shared mem buffer is in different kernels is
  //due to stupid limitations in cuda :)
  extern __shared__ __half2 shared_buffer4[];
  __half2* input_buffer = shared_buffer4;
  __half2* output_buffer = shared_buffer4 + 4096;

  #pragma unroll
  for(int k=0; k<8; k++){
    int i = threadIdx.x + (512 * k);
    input_buffer[i] = input_data[i];
  }

  #pragma unroll
  for(int k=0; k<8; k++){
    int i = threadIdx.x + (512 * k);
    int j = (i % 16) * 256 + (((i / 16) % 16) * 16) + (i / 256);

    output_buffer[j] = input_buffer[i];
  }

  #pragma unroll
  for(int k=0; k<8; k++){
    int i = threadIdx.x + (512 * k);
    output_data[i] = output_buffer[i];
  }
}

template <typename Integer>
__global__ void ShortTransposer(__half2* input_data, __half2* output_data,
                                Integer fft_length, int amount_of_r2_steps){

}
