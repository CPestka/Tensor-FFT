//Contains the kernel that performs all needed transpose operations on the fft
//input data
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

//This kernel performs the reordering of the input data so that the neccessary
//input data for the subsequent kernels (i.e. DFTKernel(), Radix2Kernel() and
//Radix16Kernel()) is continuous in memory.
//The radix2 algorithm combines two ffts of length N/2 of the even and odd
//elements to a length N fft. To have the odds and evens continuously arranged
//in memory one can reinterprete the length N array as a 2xN/2 array (no data is
//moved only the accessing semantics change) and then transpose the matrix (now
//data is actualy moved). The radix 16 algorithm works analogously but operates
//on 16 N/16 point ffts instead and requires a 16xN/16 matrix transpose.
//The algorithms combine a fixed amount of smaller ffts to larger ones. This
//means that they can be applied recursively until the ffts which they combine
//are either of size 1, in which case the DFT(x)=x, or the smaller ffts are
//computed by direct multiplication with the dft matrix (which is done in this
//implementation for ffts of size 16 by the kernel DFTKernel()). This recursive
//usage of the algorithm requires numerous transpose operations on different
//regions of the data.
//This kernel does not sequentialy perform these transposes. Instead each thread
//computes for one index of the input array the index in the output array after
//ALL transposes and then performs the copy from the input to the output array
//for that element. This reduces the amount of global memory read-writes from
//fft_length*(amount_of_radix16_steps + amount_of_radix2_steps) to
//fft_length.
//Example: fft_length=N=2*16*16*16=8192 -> with position in inital array x[i]
//the position in the output array is then y[i%2][(i/2)%16]][((i/2)/16)%16]
//[(((i/2)/16)/16)/16]. The according memory offset can be computed from those
//indecies y[id_0][id_1]... In this case -> output_offset = (N/2)*id_0 +
//(N/32)*id_1 + (N/512)*id_2 + (N/8192)*id_3
__global__ void TransposeKernel(__half* input_data_RE, __half* input_data_IM,
                                __half* output_data_RE, __half* output_data_IM,
                                int fft_length, int amount_of_r16_steps,
                                int amount_of_r2_steps) {
  //The thread id is the id for the entry of the input array we wish to store to
  //the correct position in the output array
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < fft_length) { //Check if entry within bounds
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
    output_data_RE[output_id] = input_data_RE[id];
    output_data_IM[output_id] = input_data_IM[id];
  }
}
