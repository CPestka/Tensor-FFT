//This file contains the kernel TensorFFT256 that performs the "baselayer" FFTs
//up to the size of 256. This is done by performing the "transposes" on the
//input data, performing a length 16 DFT on that and than a radix16 step on
//that.
//The bundleing of these task allows the results and auxilary data to be kept in
//loacl/shared memory.
#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "DFTMatrix.h"

using namespace nvcuda;

template <typename Integer>
__global__ void TensorFFT256(__half* input_data_RE, __half* input_data_IM,
                             __half* output_data_RE, __half* output_data_IM,
                             Integer fft_length, int amount_of_r16_steps,
                             int amount_of_r2_steps){
  Integer thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  Integer warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  int inter_block_warp_id = warp_id % (blockDim.x / 32);
  //Used to devide work for threads in a warp since the problem size is 16 based
  //and the tensor core operations are "warp wide".
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  //4 dynamic shared memory buffers
   extern __shared__ __half buffer[];
   int warp_shared_memory_offset = 1024 * inter_block_warp_id;
   Integer warp_global_memory_offset = 256 * warp_id;
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

  //On the fly computation of DFT matrix
  //TODO: test speed and accuracy of cos,cosf,coh (and modulo version of those)
  //and literal version
  // #pragma unroll
  // for(int k=0; k<8; k++){
  //   int j = k + 8 * inter_warp_id_is_upper_16;
  //   int buffer_array_id = inter_warp_id_16 + 16 * j;
  //
  //   __half phase =
  //       __hdiv(__hmul(static_cast<__half>(j * inter_warp_id_16),
  //                     static_cast<__half>(M_PI)),
  //              static_cast<__half>(8.0));
  //   //float phase = (static_cast<float>(j * inter_warp_id_16) * M_PI) / 8.0;
  //
  //   buffer_RE[buffer_array_id] = hcos(phase);
  //   buffer_IM[buffer_array_id] = -hsin(phase);
  // }

  //Literal version of dft matrix.
  LoadLiteralDFTMatrixToShared(inter_warp_id, buffer_RE, buffer_IM);

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
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0);
  wmma::fill_fragment(accumulator_IM_frag, 0.0);

  //Compute x of input_data[x] for a given "shuffeld"_input_data[thread_id],
  //where the "shuffeld" version referres to the data being reorded in the
  //following way.
  //We perform the reordering of the input data so that the neccessary
  //input data for the subsequent computation and kernels is continuous in
  //memory.
  //The radix2 algorithm combines two ffts of length N/2 of the even and odd
  //elements to a length N fft. To have the odds and evens continuously arranged
  //in memory one can reinterprete the length N array as a 2xN/2 array (no data
  //is moved only the accessing semantics change) and then transpose the matrix
  //(now data is moved). The radix 16 algorithm works analogously but operates
  //on 16 N/16 point ffts instead and requires a 16xN/16 matrix transpose.
  //The algorithms combine a fixed amount of smaller ffts to larger ones. This
  //means that they can be applied recursively until the ffts which they combine
  //are either of size 1, in which case the DFT(x)=x, or the smaller ffts are
  //computed by direct multiplication with the dft matrix (which is done in this
  //implementation for ffts of size 16. This recursive usage of the algorithm
  //requires numerous transpose operations on different regions of the data.
  //This kernel does not sequentialy perform these transposes. Instead each
  //thread computes for one index of the input array the index in the output
  //array after ALL transposes and then performs the copy from the input to the
  //output array for that element. This reduces the amount of global memory
  //read-writes from fft_length*(amount_of_radix16_steps +
  //amount_of_radix2_steps) to fft_length.
  //Example: fft_length=N=2*16*16*16=8192 -> with position in inital array x[i]
  //the position in the output array is then y[i%2][(i/2)%16]][((i/2)/16)%16]
  //[(((i/2)/16)/16)/16]. The according memory offset can be computed from those
  //indecies y[id_0][id_1]... In this case -> output_offset = (N/2)*id_0 +
  //(N/32)*id_1 + (N/512)*id_2 + (N/8192)*id_3
  #pragma unroll
  for(int k=0; k<8; k++){
    int buffer_array_id = inter_warp_id_16 +
                          16 * (k + 8 * inter_warp_id_is_upper_16);
    Integer output_data_id = buffer_array_id + warp_global_memory_offset;

    Integer tmp_id = output_data_id;

    Integer input_array_id = 16 * (tmp_id % 16);
    tmp_id = tmp_id / 16;

    input_array_id += (tmp_id % 16);
    //At the moment input_array_id =
    //16 * (output_data_id % 16) + (output_data_id / 16) % 16

    if (amount_of_r2_steps == 0) {
      for(int i=1; i<amount_of_r16_steps; i++){
        tmp_id = tmp_id / 16;
        input_array_id = (16 * input_array_id) + (tmp_id % 16);
      }
    } else {
      for(int i=1; i<amount_of_r16_steps; i++){
        tmp_id = tmp_id / 16;
        input_array_id = (16 * input_array_id) + (tmp_id % 16);
      }

      //First step of new dimension size is different
      tmp_id = tmp_id / 16; // <--
      input_array_id = (2 * input_array_id) + (tmp_id % 2);

      //Further steps is needed
      for(int i=1; i<amount_of_r2_steps; i++){
        tmp_id = tmp_id / 2; //<--
        input_array_id = (2 * input_array_id) + (tmp_id % 2);
      }
    }


    //For unscaled results
    // buffer_RE[buffer_array_id] = input_data_RE[input_array_id];
    // buffer_IM[buffer_array_id] = input_data_IM[input_array_id];

    //For sequential scaling
    buffer_RE[buffer_array_id] = __hdiv(input_data_RE[input_array_id],
                                       static_cast<__half>(256.0));
    buffer_IM[buffer_array_id] = __hdiv(input_data_IM[input_array_id],
                                       static_cast<__half>(256.0));

    //For scaling in one step
    // buffer_RE[buffer_array_id] = __hdiv(input_data_RE[input_array_id],
    //                                    static_cast<__half>(fft_length));
    // buffer_IM[buffer_array_id] = __hdiv(input_data_IM[input_array_id],
    //                                    static_cast<__half>(fft_length));
  }

  //Load the inputs
  wmma::load_matrix_sync(data_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(data_IM_frag, buffer_IM, 16);

  //
  //Perform DFT of length 16 via multiplication with DFT matrix
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
  wmma::store_matrix_sync(buffer_RE, accumulator_RE_1_frag, 16,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(buffer_tmp_RE, accumulator_RE_2_frag, 16,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(buffer_IM, accumulator_IM_frag, 16,
                          wmma::mem_row_major);

  //RE = RE_1 + RE_2, IM = IM_1 + IM_2
  #pragma unroll
  for(int k=0; k<8; k++){
    int buffer_array_id = inter_warp_id_16 +
                          (16 *  (k + (8 * inter_warp_id_is_upper_16)));

    buffer_RE[buffer_array_id] -= buffer_tmp_RE[buffer_array_id];
  }

  //
  //Perform first R16 step
  //

  //TODO: Not sure if transpose is  better than using new fragments
  //Load 16 16-point ffts from shared mem buffer, multiply them with according
  //twiddle factors and store them to other shared memory buffer. During that
  //a transpose is also performed.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    int buffer_array_id = (inter_warp_id_16 + 16 * j);
    int buffer_array_id_transposed = (j + 16 * inter_warp_id_16);

    //On the fly computation of DFT matrix
    //TODO: test speed and accuracy of cos,cosf,coh (and modulo version of those)
    //and literal version
    __half phase =
        __hdiv(__hmul(static_cast<__half>(inter_warp_id_16 * j),
                      static_cast<__half>(M_PI)),
               static_cast<__half>(128.0));
    float phase = (static_cast<float>(inter_warp_id_16 * j) * M_PI) / 128.0;

    __half twiddle_RE = hcos(phase);
    __half twiddle_IM = -hsin(phase);

    __half input_RE = buffer_RE[buffer_array_id];
    __half input_IM = buffer_IM[buffer_array_id];

    //Store modified data to buffer arrays (buffer needed due to transpose)
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

  //Store the output
  wmma::store_matrix_sync(buffer_RE, accumulator_RE_1_frag, 16,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(buffer_tmp_RE, accumulator_RE_2_frag, 16,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(buffer_IM, accumulator_IM_frag, 16,
                          wmma::mem_row_major);

  //RE = RE_1 + RE_2, IM = IM_1 + IM_2
  #pragma unroll
  for(int k=0; k<8; k++){
    int buffer_array_id = inter_warp_id_16 +
                          (16 *  (k + (8 * inter_warp_id_is_upper_16)));

    buffer_RE[buffer_array_id] -= buffer_tmp_RE[buffer_array_id];
  }

  //Store results into global memory and revert transpose.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    int buffer_array_id_transposed = j + (16 * inter_warp_id_16);
    //Global id also reverses the transpose
    Integer global_array_id = inter_warp_id_16 + (16 * j) +
                              warp_global_memory_offset;

    output_data_RE[global_array_id] = buffer_RE[buffer_array_id_transposed];
    output_data_IM[global_array_id] = buffer_IM[buffer_array_id_transposed];
  }
}
