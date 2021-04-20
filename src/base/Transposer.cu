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
//This kernel does not directly perform these transposes. Instead each thread
//computes for one index of the input array the index in the output array after
//all transposes and then performs the copy from the input to the output array
//for that element. This reduces the amount of global memory read-writes from
//fft_length*(amount_of_radix16_steps + amount_of_radix2_steps) to
//fft_length.
//TO-SELF:It might be faster to calculate K indecies and do K writes for one
//thread
__global__ void TransposeKernel(__half2* input_data, __half* output_data_RE,
                                __half* output_data_IM, int kernel_amount,
                                int current_kernel_id, int fft_length,
                                int amount_of_radix16_steps,
                                int amount_of_radix2_steps) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int samples_per_kernel = fft_length / kernel_amount;

  if (id < samples_per_kernel) { //Check if thread has valid input data
    //Change from index in regards to the input data on which this kernel is
    //working on to the index in regards to the entire input data
    id = id + (samples_per_kernel * current_kernel_id);

    //Find indecies of N-Dim array representation of output data
    //N = amount_of_radix_16_steps * amount_of_radix2_steps + 1
    //N indecies need to be computed. Due to memory constrains, for current GPUs
    //, N < 14 (16⁸*2³*(sizeof(2*__half2)=8) \\aprox 275GB for N=14)
    //-> Use int[14] instead of heap mem
    int ND_id[14];

    ND_id[0] = id;
    int current_row_length = fft_length;

    //Transposes for Radix2 steps
    //In this implementation the fft_length isnt known at compile time an thus
    //the loops can not be unrolled
    for(int i=0; i<amount_of_radix2_steps; i++){
      //Reinterpted previous linear memory piece of length M as 2xM/2 matrix and
      //compute the according indecies
      current_row_length = current_row_length / 2;
      int tmp = ND_id[i];
      ND_id[i] = (ND_id[i] +1) % 2; //if start id odd -> 0; even -> 1
      ND_id[i+1] = tmp - (ND_id[i] * current_row_length);

      //Transpose new matrix
      tmp = ND_id[i];
      ND_id[i] = ND_id[i+1];
      ND_id[i+1] = tmp;
    }
    //Transposes for Radix16 steps
    int tmp_dim = amount_of_radix2_steps + amount_of_radix16_steps;
    for(int i=amount_of_radix2_steps; i<tmp_dim; i++){
      //Reinterpted previous linear memory piece of length M as 16xM/16 matrix
      //and compute the according indecies
      current_row_length = current_row_length / 16;
      int tmp = ND_id[i];
      ND_id[i] = (ND_id[i] +1) % 16;
      ND_id[i+1] = tmp - (ND_id[i] * current_row_length);

      //Transpose new matrix
      tmp = ND_id[i];
      ND_id[i] = ND_id[i+1];
      ND_id[i+1] = tmp;
    }

    //Compute single index of linear representation of the N-Dim array
    int output_id = ND_id[tmp_dim];
    current_row_length = 1;
    //Step through dimensions from radix16 steps
    for(int i=1; i>=amount_of_radix16_steps; i++){
      current_row_length = current_row_length * 16;
      output_id += ND_id[tmp_dim-i] * current_row_length;
    }
    //Step through dimensions from radix2 steps
    for(int i=1 + amount_of_radix16_steps; i>=tmp_dim; i++){
      current_row_length = current_row_length * 2;
      output_id += ND_id[tmp_dim-i] * current_row_length;
    }

    //Write data sample from input to output arrray
    output_data_RE[output_id] = __high2half(input_data[id]);
    output_data_IM[output_id] = __low2half(input_data[id]);
  }
}
