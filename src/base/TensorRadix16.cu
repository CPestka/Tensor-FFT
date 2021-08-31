//Contains the kernel that performs the radix16 steps on tensor cores
#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "DFTMatrix.h"

using namespace nvcuda;

//This kernel performs the radix 16 recombination steps. It takes 16 length
//N/(16^m) with m being the number of already performed radix 16 steps + 1 and
//combines them to a 16^(m+1) length FFT. Multiple of these results are then
//used as the input for another call of this kernel or the Radix2Kernel() untill
//the final FFT length is reached.
//The Kernel can be divided into 3 main sections. Contrary to the first steps in
//the e.g. TensorFFT256 kernel, the input data cant just be read and used
//directly by the tensor cores. Instead a componentwise multiplication with the
//so called twiddle factors has to be performed first. Due to this in the first
//section each warp loads the input data, computes the multiplication and stores
//the result in its own section of a shared memory buffer.
//In the second and third section the data is then loaded into the fragments
//and the matrix multiplication of the 16^mx16 data matrix with the 16x16 DFT
//matrix is performed. For m > 1 the matrix multiplication is split into m
//16x16 * 16x16 matrixmultiplications and the results are then recombined
//by storing the results in the correct place in memory. Also due to this the
//input data for m > 1 isnt linear in memory but for one 16x16 matrix instead
//we have 16 linear chuncks of length 16 that are each offset to each other by
//sub_fft_length=16^(m+1).
//This kernel is, if the fft legth is long enough, called (multiple times) after
//either TensorFFT256 or TensorFFT4096 and then followed Radix2Kernel.
template <typename Integer>
__global__ void TensorRadix16(__half* input_data_RE, __half* input_data_IM,
                              __half* output_data_RE, __half* output_data_IM,
                              Integer fft_length, Integer sub_fft_length) {
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
   int warp_shared_memory_offset = 768 * inter_block_warp_id;
   __half* buffer_RE = buffer + warp_shared_memory_offset;
   __half* buffer_IM = buffer + warp_shared_memory_offset + 256;
   __half* buffer_tmp_RE = buffer + warp_shared_memory_offset + 512;

  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;

  //On the fly computation of DFT matrix
  //TODO: test speed and accuracy of cos,cosf,coh (and modulo version of those)
  //and literal version
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + 8 * inter_warp_id_is_upper_16;
    int buffer_array_id = inter_warp_id_16 + 16 * j;

    // __half phase =
    //     __hdiv(__hmul(static_cast<__half>(j * inter_warp_id_16),
    //                   static_cast<__half>(M_PI)),
    //            static_cast<__half>(8.0));
    float phase = (static_cast<float>(j * inter_warp_id_16) * M_PI) / 8.0;

    buffer_RE[buffer_array_id] = cos(phase);
    buffer_IM[buffer_array_id] = -sin(phase);
  }

  //Literal version of dft matrix.
  LoadLiteralDFTMatrixToShared(inter_warp_id, buffer_RE, buffer_IM);

  //Load DFT matrix into the according fragments
  wmma::load_matrix_sync(dft_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(dft_IM_frag, buffer_IM, 16);

  Integer combined_fft_length = sub_fft_length * 16;
  Integer amount_of_warps_pes_substep = sub_fft_length / 16;
  Integer inter_substep_id = warp_id % amount_of_warps_pes_substep;
  Integer substep_id = warp_id / amount_of_warps_pes_substep;

  //Each of the 32 threads pre warp loads 8 (8*32=16*16) data
  //points. However the data of the needed 16x16 matrix of input data is not
  //linaer in memory. The entire 16^mx16 matrix (which is linear in memory) is
  //divided into m 16x16 matrices. This means that the data for one 16x16
  //matrix consists of 16 length 16 linear chuncks, which are offset in
  //respect to each other by sub_fft_length=16^m.
  //Reads are linear in inter_warp_id_16 -> Global reads of 32bytes of
  //sequential memory. Smallest "cuda read" is 32 bytes -> no waste
  //But TODO: ? version that takes 8 matrices -> 256bytes = largest "cuda read"
  //Also, by swaping the indecies when loading the storing to and from the
  //fragment the fragment holds the transposed data, which is needed since the
  //data is stored in row major order in memory but is needed in collum major
  //for the matrix multiplication.
  #pragma unroll
  for(int k=0; k<8; k++){
    Integer i = inter_warp_id_16 + (inter_substep_id * 16);
    int j = k + (8 * inter_warp_id_is_upper_16);
    Integer global_memory_offset = i +
                                   sub_fft_length * j +
                                   substep_id * combined_fft_length;
    int buffer_matrix_memory_offset = j + 16 * inter_warp_id_16;

    //On the fly computation of twiddle fctors
    //TODO: test speed and accuracy of cos,cosf,coh (and modulo version of those)
    //and literal version (look up table of N points cos(2*PI*i/N ))
    //Float because static_cast<__half>(combined_fft_length) overflows
    float tmp = static_cast<float>(i * j)  /
                static_cast<float>(combined_fft_length);
    // __half phase = __hmul(__hmul(2.0, static_cast<__half>(M_PI)),
    //                       static_cast<__half>(tmp));
    float phase = 2.0 * M_PI * tmp;

    //TO-SELF: test __cosf vs cos accuracy and speed
    __half twiddle_RE = cos(phase);
    __half twiddle_IM = -sin(phase);

    //Fetch current data once from global memory to use it twice
    //For unscaled or scaling at once
    // __half input_RE = input_data_RE[global_memory_offset];
    // __half input_IM = input_data_IM[global_memory_offset];

    //For sequential scaling
    __half input_RE = __hdiv(input_data_RE[global_memory_offset],
                            static_cast<__half>(16.0));
    __half input_IM = __hdiv(input_data_IM[global_memory_offset],
                            static_cast<__half>(16.0));

    //Store modified data to buffer arrays
    //mod_RE = RE*twid_RE - IM*twid_IM
    buffer_RE[buffer_matrix_memory_offset] =
        __hsub(__hmul(input_RE, twiddle_RE), __hmul(input_IM, twiddle_IM));
    //mod_IM = RE*twid_IM + IM*twid_RE
    buffer_IM[buffer_matrix_memory_offset] =
        __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
  }

  //Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      data_RE_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      data_IM_frag;

  //Load the modified data from shared mem buffer
  wmma::load_matrix_sync(data_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(data_IM_frag, buffer_IM, 16);

  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_2_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_IM_frag;

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
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

  //Store the results in the appropriately reordered way into the output array
  //The data is stored back the way it was intialy the i.e. a 16^mx16 linear=
  //row-major array and is then reinterpreted as a linear in memory FFT of
  //length 16^(m+1)
  //The transpose operation is also reverted.
  #pragma unroll
  for(int k=0; k<8; k++){
    Integer i = inter_warp_id_16 + (inter_substep_id * 16);
    int j = k + (8 * inter_warp_id_is_upper_16);
    Integer global_memory_offset = i +
                                   (sub_fft_length * j) +
                                   (substep_id * combined_fft_length);
    int buffer_matrix_memory_offset = j + 16 * inter_warp_id_16;

    output_data_RE[global_memory_offset] =
        buffer_RE[buffer_matrix_memory_offset];
    output_data_IM[global_memory_offset] =
        buffer_IM[buffer_matrix_memory_offset];
  }
}
