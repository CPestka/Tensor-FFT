//This file contains the kernel TensorFFT4096 that performs the "baselayer" FFTs
//up to the size of 4096. It takes in the shuffled data, that was prepared by
//the on the Transposer() kernel and performes a length 16 DFT on that and
//than two radix16 steps on that.
//The bundleing of these task allows the results and auxilary data to be kept in
//loacl/shared memory.
//This kernel HAS TO BE LAUNCHED WITH EXACTLY 16 warps per thread i.e.
//blocksize = 16*32 = 512
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template <typename Integer>
__global__ void TensorFFT4096(__half2* input_data,
                              __half2* output_data,
                              Integer fft_length,
                              int amount_of_r16_steps,
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
  extern __shared__ __half shared_buffer2[];
  int warp_shared_memory_offset = 1024 * inter_block_warp_id;
  Integer warp_global_memory_offset = 256 * warp_id;
  __half* buffer_RE = shared_buffer2 + warp_shared_memory_offset;
  __half* buffer_IM = shared_buffer2 + warp_shared_memory_offset + 256;
  __half* buffer_tmp_RE = shared_buffer2 + warp_shared_memory_offset + 512;
  __half* buffer_tmp_IM = shared_buffer2 + warp_shared_memory_offset + 768;

  //
  //Setup DFT Matrix
  //

  //Declare fragments that will hold the DFT matrix
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;

  //On the fly computation of DFT matrix
  //TODO: test speed and accuracy of cos,cosf,coh and literal version
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + 8 * inter_warp_id_is_upper_16;
    int buffer_array_id = inter_warp_id_16 + 16 * j;

    // __half phase =
    //     __hdiv(__hmul(static_cast<__half>(j * inter_warp_id_16),
    //                   static_cast<__half>(M_PI)),
    //            static_cast<__half>(8.0));
    float phase = (static_cast<float>(j * inter_warp_id_16) * M_PI) / 8.0;

    buffer_RE[buffer_array_id] = cosf(phase);
    buffer_IM[buffer_array_id] = -sinf(phase);
  }

  //Literal version of dft matrix.
  //LoadLiteralDFTMatrixToShared(inter_warp_id, buffer_RE, buffer_IM);

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

  #pragma unroll
  for(int k=0; k<8; k++){
    int buffer_array_id = inter_warp_id + 32*k;
    //Fetch packed data point from global memory
    __half2 tmp = input_data[buffer_array_id + warp_global_memory_offset];

    //Unpack, apply sequential scaling and save to shared mem
    buffer_RE[buffer_array_id] = __hdiv(tmp.x, static_cast<__half>(4096.0));
    buffer_IM[buffer_array_id] = __hdiv(tmp.y, static_cast<__half>(4096.0));
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

  //RE = RE_1 - RE_2, IM = IM_1 + IM_2
  #pragma unroll
  for(int k=0; k<8; k++){
    int buffer_array_id = inter_warp_id_16 +
                          (16 *  (k + (8 * inter_warp_id_is_upper_16)));

    buffer_RE[buffer_array_id] -= buffer_tmp_RE[buffer_array_id];
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

    //On the fly computation of DFT matrix
    //TODO: test speed and accuracy of cos,cosf,coh (and modulo version of those)
    //and literal version
    // __half phase =
    //     __hdiv(__hmul(static_cast<__half>(inter_warp_id_16 * j),
    //                   static_cast<__half>(M_PI)),
    //            static_cast<__half>(128.0));
    float phase = (static_cast<float>(inter_warp_id_16 * j) * M_PI) / 128.0;

    __half twiddle_RE = cosf(phase);
    __half twiddle_IM = -sinf(phase);

    __half input_RE = buffer_RE[buffer_array_id];
    __half input_IM = buffer_IM[buffer_array_id];

    //Store modified data to buffer arrays (buffer needed due to transpose)
    //TODO: ? remove second buffer and use collum major load better?
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
  //Perform second R16 step
  //

  //For the second r16 step the results of 16 other warps are needed to perform
  //one 4096 length combine. This should fit in shared memory on most devcices
  //however due to a growth of 16x with each step in shared mem usage this isnt
  //possible for further steps atm.static_cast<__half>(
  //Reorder and multiply with twiddle factors the results of the first r16 step
  //so the matrix to load in a fragment is linear in memory (see TensorRadix16
  //kernel for why).
  //Indexing used takes care of the following steps: revert transpose of results
  //, interprete result 16x16 matrix as one of 16 rows of 256x16 a matrix with
  //row_id=inter_block_warp_id, safe 16  16x16 matrix cut out of that matrix in
  //buffer_tmp and transpose the entries.
  //Each warp only fills 1/16 of a needed matrix and thus stores 16 entries each
  //in the buffer_tmp section of ALL 16 warps. The tmp buffer is needed by the
  //previous step -> wait for all 16 warps in this block
  __syncthreads();
  #pragma unroll
  for(int k=0; k<8; k++){
    //int i = inter_warp_id_16;
    int j = k + (8 * inter_warp_id_is_upper_16);
    //For correct phase
    int i_global = j + (16 * inter_warp_id_16);

    int buffer_array_id_old = inter_warp_id_16 + (16 * j);
    int buffer_array_id_new = inter_block_warp_id +
                              (16 * j) +
                              (1024 * inter_warp_id_16);

    //On the fly computation of DFT matrix
    //TODO: test speed and accuracy of cos,cosf,coh and literal version
    // __half phase = (static_cast<float>(i_global * inter_block_warp_id) * M_PI) /
    //                static_cast<float>(2048.0);
    float phase = (static_cast<float>(i_global * inter_block_warp_id) * M_PI) /
                  2048.0;

    __half twiddle_RE = cosf(phase);
    __half twiddle_IM = -sinf(phase);

    __half input_RE = buffer_RE[buffer_array_id_old];
    __half input_IM = buffer_IM[buffer_array_id_old];

    //Store modified data to buffer_tmp arrays
    //The 1024*j of buffer_array_id_new "selects" whoes warps shared mem buffer
    //to write to and the +512 / +768 selects the tmp_RE / tmp_IM section
    //TODO: ? remove second buffer and use collum major load better?
    //mod_RE = RE*twid_RE - IM*twid_IM
    shared_buffer2[buffer_array_id_new + 512] =
        __hsub(__hmul(input_RE, twiddle_RE), __hmul(input_IM, twiddle_IM));

    //mod_IM = RE*twid_IM + IM*twid_RE
    shared_buffer2[buffer_array_id_new + 768] =
        __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
  }

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0);
  wmma::fill_fragment(accumulator_IM_frag, 0.0);

  //Wait for the other 15 rows of 16 elements from other warps within block
  __syncthreads();

  //Load the modified data from shared mem buffer
  wmma::load_matrix_sync(data_RE_frag, buffer_tmp_RE, 16);
  wmma::load_matrix_sync(data_IM_frag, buffer_tmp_IM, 16);

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
                          (16 * (k + (8 * inter_warp_id_is_upper_16)));

    buffer_RE[buffer_array_id] -= buffer_tmp_RE[buffer_array_id];
  }

  //Store the results in the appropriately reordered way into the output array
  //The data is stored back the way it was intialy the i.e. a 16^mx16 linear=
  //row-major array and is then reinterpreted as a linear in memory FFT of
  //length 16^(m+1)
  //The transpose operation is also reverted.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    Integer global_memory_offset = inter_warp_id_16 +
                                   (16 * inter_block_warp_id) +
                                   (256 * j) +
                                   (4096 * (warp_id / 16));
    int buffer_matrix_memory_offset = j + (16 * inter_warp_id_16);

    __half2 tmp;
    tmp.x = buffer_RE[buffer_matrix_memory_offset];
    tmp.y = buffer_IM[buffer_matrix_memory_offset];

    output_data[global_memory_offset] = tmp;
  }
}
