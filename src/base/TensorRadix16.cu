//Contains the kernel that performs the radix16 steps on tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

//This kernel performs the radix 16 recombination steps. It takes 16 length
//N/(16^m) with m being the number of already performed radix 16 steps + 1 and
//combines them to a 16^(m+1) length FFt. These are then used as the input for
//another call of this kernel or the Radix2Kernel() untill the final FFT length
//is reached.
//The Kernel can be divided into 3 main sections. Contrary to the DFTKernel()
//the input data cant just be read and used directly by the tensor cores.
//Instead a componentwise multiplication with the so called twiddle factors has
//to be performed first. Due to this in the first section each warp loads the
//input data computes the multiplication and stored the result in its own
//section of a shared memory buffer.
//In the second and third section the data is then loaded into the fragments
//and the matrix multiplöication of the 16^mx16 data matrix with the 16x16 DFT
//matrix is performed. For m > 1 the matrix multiplication is split into m
//16x16 * 16x16 matrixmultiplications and the results are then recombined in
//by storing the results in the correct place in memory. Also due to this the
//input data for m > 1 isnt linaer in memory but for one 16x16 matrix instead
//16 offset linear chuncks of length 16.
//The needed batching of the matrix multiplications for the tensor cores is (16
//16x16 matrices have to be qued at once) is handeled for the m = 1 case by
//performing 16 of the length 16 -> length 256 combines at once and for the case
//m > 1 via batching multiple of the 16x16 multiplications of one or more
//recombine steps.
__global__ void Radix16Kernel(__half* input_data_RE, __half* input_data_IM,
                              __half* output_data_RE, __half* output_data_IM,
                              __half* dft_matrix_batch_RE,
                              __half* dft_matrix_batch_IM,
                              int kernel_amount, int current_kernel_id,
                              int fft_length, int current_radix16_step) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  //Used to devide work for threads in a warp since the problem size is 16 based
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  int memory_kernel_offset =
      (fft_length / kernel_amount) * current_kernel_id;
  int warp_memory_offset = memory_kernel_offset + (4096 * warp_id);
  
  int sub_fft_length = (current_radix16_step + 1) * 16;
  int combined_fft_length = sub_fft_length * 16;

  //TO-DO: Use direct acces of data via fragment instead of shared memory buffer
  //Buffers used to store the input data multiplied by the twiddle factors
  //temporarily before loading them into the fragments.
  //16*16*16*2*4*2 = 64KB max mememory for these buffers per SM (4 TC per SM)
  //A100 164KB register file -> fine on that
  extern __shared__ __half shared_memory[];
  __half* modified_input_RE = shared_memory + warp_id * 8192;
  __half* modified_input_IM = shared_memory + warp_id * 8192 + 4096;

  //In this case one warp performs 16 combines of 16 size 16 FFts. This means
  //that the resulting data does not need to be rearanged.
  //Each of the 32 threads pre warp loads 8*16=128 (128*32=16*16*16) data points
  if (current_radix16_step == 0) {
    #pragma unroll
    for(int i=0; i<8; i++){
      #pragma unroll
      for(int j=0; j<16; j++){
        int matrix_memory_offset = inter_warp_id_16 + (16 * j) +
            (256 * (i + (8 * inter_warp_id_is_upper_16)));
        int total_memory_offset = warp_memory_offset + matrix_memory_offset;

        //Compute RE and IM of twiddle factors
        float phase = (-2 * M_PI * j * inter_warp_id_16) /
                        combined_fft_length;
        //TO-SELF: test __cosf vs cos accuracy and speed
        __half twiddle_RE = __float2half(cosf(phase));
        __half twiddle_IM = __float2half(sinf(phase));

        //Fetch current data once from global memory to use it twice
        __half input_RE = input_data_RE[total_memory_offset];
        __half input_IM = input_data_IM[total_memory_offset];

        //Store modified data to buffer arrays
        modified_input_RE[matrix_memory_offset] =
            __hfma(input_RE , twiddle_RE,
                   __hmul(-1.0, __hmul(input_IM, twiddle_IM)));
        modified_input_IM[matrix_memory_offset] =
            __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
      }
    }

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
    wmma::load_matrix_sync(dft_RE_frag, dft_matrix_batch_RE, 16);
    wmma::load_matrix_sync(dft_IM_frag, dft_matrix_batch_IM, 16);
    wmma::load_matrix_sync(data_RE_frag, modified_input_RE, 16);
    wmma::load_matrix_sync(data_IM_frag, modified_input_IM, 16);

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

    //Store IM part of the output
    wmma::store_matrix_sync(output_data_IM + warp_memory_offset,
                            accumulator_IM_frag, 16, wmma::mem_row_major);

    //RE(A)xRE(B) - IM(A)xIM(B) has to be performed which cant be done directly
    //with the fragments. Instead temporarily save the results in 2 buffer arrays,
    //compute the differnce and store it. For this purpose the no longer needed
    //data arrays modified_input_x are reused.
    wmma::store_matrix_sync(modified_input_RE, accumulator_RE_1_frag, 16,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(modified_input_IM, accumulator_RE_2_frag, 16,
                            wmma::mem_row_major);

    //Each thread in a warp computes 128 values and stores them -> 32*128=4096=
    //16*16*16
    #pragma unroll
    for(int i=0; i<128; i++){
      int current_id = inter_warp_id * 128 + i;
      output_data_RE[warp_memory_offset + current_id] =
          __hsub(modified_input_RE[current_id], modified_input_IM[current_id]);
    }
  } else { //case m > 1
    int amount_of_substeps = sub_fft_length / 16;
    int inter_substep_id = warp_id % amount_of_substeps;

    //Each of the 32 threads pre warp loads 8*16=128 (128*32=16*16*16) data
    //points. However the data of the needed 16x16 matrix of input data is not
    //linaer in memory. The entire 16^mx16 matrix (which is linear in memory) is
    //divided into m 16x16 matrices. This means that the data for one 16x16
    //matrix consists of 16 length 16 linear chuncks, which are offset in
    //respect to each other by sub_fft_length=16^m.
    #pragma unroll
    for(int i=0; i<8; i++){
      #pragma unroll
      for(int j=0; j<16; j++){
        int matrix_memory_offset =
            inter_warp_id_16 +
            (sub_fft_length * j) +
            (16 * (i + (8 * inter_warp_id_is_upper_16))) +
            (16 * 16 * warp_id);
        int total_memory_offset = warp_memory_offset + matrix_memory_offset;

        //Compute twiddle factors
        float phase = (-2 * M_PI * j * inter_substep_id) / combined_fft_length;
        //TO-SELF: test __cosf vs cos accuracy and speed
        __half twiddle_RE = __float2half(cosf(phase));
        __half twiddle_IM = __float2half(sinf(phase));

        //Fetch current data once from global memory to use it twice
        __half input_RE = input_data_RE[total_memory_offset];
        __half input_IM = input_data_IM[total_memory_offset];

        //Save modified data to buffer arrays
        modified_input_RE[matrix_memory_offset] =
            (input_RE * twiddle_RE) - (input_IM * twiddle_IM);
        modified_input_IM[matrix_memory_offset] =
            (input_RE * twiddle_IM) + (input_IM * twiddle_RE);
      }
    }

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
    wmma::load_matrix_sync(dft_RE_frag, dft_matrix_batch_RE, 16);
    wmma::load_matrix_sync(dft_IM_frag, dft_matrix_batch_IM, 16);
    wmma::load_matrix_sync(data_RE_frag, modified_input_RE, 16);
    wmma::load_matrix_sync(data_IM_frag, modified_input_IM, 16);

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

    //Store IM part to buffer. The array modified_input_RE is no longer needed
    //and can thus be used for this purpose.
    wmma::store_matrix_sync(modified_input_RE, accumulator_IM_frag, 16,
                            wmma::mem_row_major);

    //Store the results in the appropriately reordered way into the output array
    //The data is stored back the way it was intialy the i.e. a 16^mx16 linear=
    //row-major array and is then reinterpreted as a linear in memory FFT of
    //length 16^(m+1)
    #pragma unroll
    for(int i=0; i<8; i++){
      #pragma unroll
      for(int j=0; j<16; j++){
        int buffer_memory_offset = inter_warp_id_16 +
                                   16 * j +
                                   256 * (i + (8 * inter_warp_id_is_upper_16));
        int results_memory_offset =
            inter_warp_id_16 +
            (sub_fft_length * j) +
            (16 * (i + (8 * inter_warp_id_is_upper_16))) +
            (16 * 16 * warp_id) +
            warp_memory_offset;

        output_data_IM[results_memory_offset] =
            modified_input_RE[buffer_memory_offset];
      }
    }

    //RE(A)xRE(B) - IM(A)xIM(B) has to be performed which cant be done directly
    //with the fragments. Instead temporarily save the results in 2 buffer arrays,
    //compute the differnce and store it in the correctly reorded way. The
    //arrays modified_input_x are no longer needed and can thus be used for this
    wmma::store_matrix_sync(modified_input_RE, accumulator_RE_1_frag, 16,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(modified_input_IM, accumulator_RE_2_frag, 16,
                            wmma::mem_row_major);

    //Store the results in the appropriately reordered way into the output array
    #pragma unroll
    for(int i=0; i<8; i++){
      #pragma unroll
      for(int j=0; j<16; j++){
        int buffer_memory_offset = inter_warp_id_16 +
                                   16 * j +
                                   256 * (i + (8 * inter_warp_id_is_upper_16));
        int results_memory_offset =
            inter_warp_id_16 +
            (sub_fft_length * j) +
            (16 * (i + (8 * inter_warp_id_is_upper_16))) +
            (16 * 16 * warp_id) +
            warp_memory_offset;

        output_data_RE[results_memory_offset] =
            modified_input_RE[buffer_memory_offset] -
            modified_input_IM[buffer_memory_offset];
      }
    }
  }
}
