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
__global__ void TransposeKernel(__half* input_data_RE, __half* input_data_IM,
                                __half* output_data_RE, __half* output_data_IM,
                                int fft_length, int amount_of_r16_steps,
                                int amount_of_r2_steps) {
  //The thread id is the id for the entry of the input array we wish to store to
  //the correct position in the output array
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < fft_length) { //Check if entry within bounds

    //Find indecies of N-Dim array representation of output data
    //N = amount_of_radix_16_steps * amount_of_radix2_steps + 1
    //N indecies need to be computed. Due to memory constrains, for current GPUs
    //, N < 14 (16⁸*2³*(sizeof(2*__half2)=8) \\aprox 275GB for N=14)
    //-> Use int[14] instead of dynamic mem
    int ND_id[14];

    if (amount_of_r2_steps != 0) {
      //Reinterprete as transposed 2D array with the size of dim 0 as 2
      //i.e. entires with even initial id in first row, with odd in second
      //the dimension with more elements of the previous step is the one with
      //the higher dim id (due to the transpose).
      //Also compute new index i.e. perform the calculation for the memory postion
      //of x[ND_id[max_dim_id]]....[ND_id[0]]
      int current_row_length = fft_length / 2;
      ND_id[0] = id % current_row_length;
      ND_id[1] = id / current_row_length;

      int output_id = ND_id[0];
      int current_id_row_length = 2;

      //Repeat the reinterprete step for each further radix2 step
      for(int i=1; i<amount_of_r2_steps; i++){
        current_row_length = current_row_length / 2;

        ND_id[i+1] = ND_id[i] / current_row_length;
        ND_id[i] = ND_id[i] % current_row_length;

        output_id += current_id_row_length * ND_id[i];
        current_id_row_length *= 2;
      }

      //Analogous to above but for the radix 16 steps -> size of first dimension
      //is 16
      int max_dim_id = amount_of_r2_steps + amount_of_r16_steps;
      for(int i=amount_of_r2_steps; i<max_dim_id; i++){
        current_row_length = current_row_length / 16;

        ND_id[i+1] = ND_id[i] / current_row_length;
        ND_id[i] = ND_id[i] % current_row_length;

        output_id += (current_id_row_length * ND_id[i]);
        current_id_row_length *= 16;
      }
      output_id += (current_id_row_length * ND_id[max_dim_id]);

      //Move input data to correct position
      output_data_RE[output_id] = input_data_RE[id];
      output_data_IM[output_id] = input_data_IM[id];
    } else {
      //Reinterprete as transposed 2D array, like in above kernel but with size 16.
      //Also compute new index i.e. perform the calculation for the memory postion
      //of x[ND_id[0]][ND_id[1]]....[ND_id[max_dim_id]]
      int current_row_length = fft_length / 16;
      ND_id[0] = id % current_row_length;
      ND_id[1] = id / current_row_length;

      int output_id = ND_id[0];
      int current_id_row_length = 16;

      //Repeat the reinterprete step for each further radix2 step
      for(int i=1; i<amount_of_r16_steps; i++){
      current_row_length = current_row_length / 16;

      ND_id[i+1] = ND_id[i] / current_row_length;
      ND_id[i] = ND_id[i] % current_row_length;

      output_id += (current_id_row_length * ND_id[i]);
      current_id_row_length *= 16;
      }
      output_id += (current_id_row_length * ND_id[amount_of_r16_steps]);

      //Move input data to correct position
      output_data_RE[output_id] = input_data_RE[id];
      output_data_IM[output_id] = input_data_IM[id];
    }
  }
}
