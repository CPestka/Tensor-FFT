//Contains the kernel that performs the radix16 steps on tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

//These kernels performs the radix 16 recombination steps. It takes 16 length
//N/(16^m) with m being the number of already performed radix 16 steps + 1 and
//combines them to a 16^(m+1) length FFT. Multiple of these results are then
//used as the input for another call of this kernel or the Radix2Kernel() untill
//the final FFT length is reached.
//The Kernel can be divided into 3 main sections. Contrary to the DFTKernel()
//the input data cant just be read and used directly by the tensor cores.
//Instead a componentwise multiplication with the so called twiddle factors has
//to be performed first. Due to this in the first section each warp loads the
//input data, computes the multiplication and stores the result in its own
//section of a shared memory buffer.
//In the second and third section the data is then loaded into the fragments
//and the matrix multiplication of the 16^mx16 data matrix with the 16x16 DFT
//matrix is performed. For m > 1 the matrix multiplication is split into m
//16x16 * 16x16 matrixmultiplications and the results are then recombined
//by storing the results in the correct place in memory. Also due to this the
//input data for m > 1 isnt linear in memory but for one 16x16 matrix instead
//we have 16 linear chuncks of length 16 that are each offset to each other by
//sub_fft_length=16^(m+1).

//Variant of the kernel described above that is only used for the first R16
//step, due to tha fact that the calculation can be stated much cheaper for the
//first step.
__global__ void Radix16KernelFirstStep(__half* input_data_RE,
                                       __half* input_data_IM,
                                       __half* output_data_RE,
                                       __half* output_data_IM,
                                       __half* dft_matrix_batch_RE,
                                       __half* dft_matrix_batch_IM) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  int inter_block_warp_id = warp_id % (blockDim.x / 32);
  //Used to devide work for threads in a warp since the problem size is 16 based
  //and the tensor core operations are "warp wide".
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  //Declare the fragments
  //The needed matrix multiplication is normaly data x dft, but since the data
  //in global memory is stored in row major but would be needed here in collum
  //major order, we instead compute: (data x dft)^T = dft^T x data^T =
  //dft x data^T because dft is symetrical.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      data_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      data_IM_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_2_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_IM_frag;

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
  wmma::fill_fragment(accumulator_IM_frag, 0.0f);

  //Load the inputs
  int warp_memory_offset = 256 * warp_id;
  wmma::load_matrix_sync(dft_RE_frag, dft_matrix_batch_RE + warp_memory_offset,
                         16);
  wmma::load_matrix_sync(dft_IM_frag, dft_matrix_batch_IM + warp_memory_offset,
                         16);

  //Since fragments can only be accessed uniformly multiplication with different
  //factors to each element isnt possible.
  //Due to this we utilize a shared memory buffer of size of the data
  //for this block -> amount_of_warps_per_block * size_of_fragment (16*16) *
  //2 (RE + IM) * sizeof(half); (blockdim.x / 32) = amount_of_warps_per_block
  //For recomended amount_of_warps_per_block=4 -> 4kB
  extern __shared__ __half buffer[];
  __half* buffer_RE = buffer + (512 * inter_block_warp_id);
  __half* buffer_IM = buffer_RE + 256;

  //In this case one warp performs a combination of 16 size 16 FFTs. This means
  //that the resulting data does not need to be rearanged.
  //Each of the 32 threads per warp loads 8 (8*32=16*16) data points, multiplies
  //them with the twiddle factors and stores the now prepared data in the
  //fragment.
  #pragma unroll
  for(int k=0; k<8; k++){
    int j = k + (8 * inter_warp_id_is_upper_16);
    int memory_offset = (inter_warp_id_16 + 16*j);
    int global_memory_offset = warp_memory_offset + memory_offset;

    //Compute RE and IM of twiddle factors
    float phase = (2 * M_PI * inter_warp_id_16 * j) / 256;
    //TO-SELF: test __cosf vs cos accuracy and speed
    __half twiddle_RE = __float2half(cos(phase));
    __half twiddle_IM = __float2half(-sin(phase));

    //Fetch current data once from global memory to use it twice
    __half input_RE = input_data_RE[global_memory_offset];
    __half input_IM = input_data_IM[global_memory_offset];

    //Store modified data to buffer arrays
    //mod_RE = RE*twid_RE - IM*twid_IM
    buffer_RE[memory_offset] =
        __hsub(__hmul(input_RE, twiddle_RE), __hmul(input_IM, twiddle_IM));

    //mod_IM = RE*twid_IM + IM*twid_RE
    buffer_IM[memory_offset] =
        __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
  }

  //Load the modified data from shared mem buffer
  wmma::load_matrix_sync(data_RE_frag, buffer_RE, 16);
  wmma::load_matrix_sync(data_IM_frag, buffer_IM, 16);

  //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
  //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
  //RE(A)xIM(B) + IM(A)xRE(B)
  wmma::mma_sync(accumulator_RE_1_frag, dft_RE_frag, data_RE_frag,
                 accumulator_RE_1_frag);
  wmma::mma_sync(accumulator_RE_2_frag, dft_IM_frag, data_IM_frag,
                 accumulator_RE_2_frag);
  wmma::mma_sync(accumulator_IM_frag, dft_IM_frag, data_RE_frag,
                 accumulator_IM_frag);
  wmma::mma_sync(accumulator_IM_frag, dft_RE_frag, data_IM_frag,
                 accumulator_IM_frag);

  //Store IM part of the output
  wmma::store_matrix_sync(output_data_IM + warp_memory_offset,
                          accumulator_IM_frag, 16, wmma::mem_row_major);

  #pragma unroll
  for(int i=0; i<accumulator_RE_1_frag.num_elements; i++){
    output_data_RE[warp_memory_offset + i] =
        __hsub(accumulator_RE_1_frag.x[i],
               accumulator_RE_2_frag.x[i]);
  }
}

//Variant of the kernel described above that is only used for the mth step with
//m>0
__global__ void Radix16Kernel(__half* input_data_RE, __half* input_data_IM,
                              __half* output_data_RE, __half* output_data_IM,
                              __half* dft_matrix_batch_RE,
                              __half* dft_matrix_batch_IM,
                              int fft_length, int sub_fft_length,
                              int current_radix16_step) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  int inter_block_warp_id = warp_id % (blockDim.x / 32);
  //Used to devide work for threads in a warp since the problem size is 16 based
  //and the tensor core operations are "warp wide".
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  //Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      data_RE_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      data_IM_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_RE_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
      dft_IM_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_2_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_IM_frag;

  //Initialize the output to zero
  wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
  wmma::fill_fragment(accumulator_IM_frag, 0.0f);

  //Load the inputs
  int warp_memory_offset = 256 * warp_id;
  wmma::load_matrix_sync(dft_RE_frag, dft_matrix_batch_RE + warp_memory_offset,
                         16);
  wmma::load_matrix_sync(dft_IM_frag, dft_matrix_batch_IM + warp_memory_offset,
                         16);

  //Since fragments can only be accessed uniformly the reordering of the results
  //when storing them back to memory can not be done directly with the fragments
  //at all.
  //Due to this purpose we utilize a shared memory buffer of size of the data
  //for this block -> amount_of_warps_per_block * size_of_fragment (16*16) *
  //2 (RE + IM) * sizeof(half); (blockdim.x / 32) = amount_of_warps_per_block
  //For recomended amount_of_warps_per_block=4 -> 4kB -> ok on A100
  extern __shared__ __half buffer[];
  __half* buffer_RE = buffer + (512 * inter_block_warp_id);
  __half* buffer_IM = buffer_RE + 256;

  int combined_fft_length = sub_fft_length * 16;
  int amount_of_warps_pes_substep = sub_fft_length / 16;
  int inter_substep_id = warp_id % amount_of_warps_pes_substep;
  int substep_id = warp_id / amount_of_warps_pes_substep;

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
    float phase = (2 * M_PI * i * j) / combined_fft_length;
    //TO-SELF: test __cosf vs cos accuracy and speed
    __half twiddle_RE = __float2half(cos(phase));
    __half twiddle_IM = __float2half(-sin(phase));

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
}
