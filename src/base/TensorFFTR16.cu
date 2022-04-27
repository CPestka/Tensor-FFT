//Contains the kernel that performs the radix16 steps on tensor cores
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

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
__global__ void TensorRadix16(__half2* input_data,
                              __half2* output_data,
                              Integer fft_length,
                              Integer sub_fft_length) {
  Integer thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  Integer warp_id = thread_id / 32;
  int inter_warp_id = thread_id % 32;
  int inter_block_warp_id = warp_id % (blockDim.x / 32);
  //Used to devide work for threads in a warp since the problem size is 16 based
  //and the tensor core operations are "warp wide".
  int inter_warp_id_16 = inter_warp_id % 16;
  int inter_warp_id_is_upper_16 = inter_warp_id / 16;

  //Buffers for the matrices
   extern __shared__ __half shared_buffer[];
   int warp_shared_memory_offset = 1536 * inter_block_warp_id;
   __half* matrix_b_dft_RE = shared_buffer + warp_shared_memory_offset;
   __half* matrix_b_dft_IM = shared_buffer + warp_shared_memory_offset + 256;
   __half* matrix_a_data_RE = shared_buffer + warp_shared_memory_offset + 512;
   __half* matrix_a_data_IM = shared_buffer + warp_shared_memory_offset + 768;
   __half* matrix_acc_RE = shared_buffer + warp_shared_memory_offset + 1024;
   __half* matrix_acc_IM = shared_buffer + warp_shared_memory_offset + 1280;

  // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
  //     dft_RE_frag;
  // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
  //     dft_IM_frag;

  //On the fly computation of DFT matrix
  //TODO: test speed and accuracy of cos,cosf,hcos
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

    matrix_b_dft_RE[buffer_array_id] = cosf(phase);
    matrix_b_dft_IM[buffer_array_id] = -sinf(phase);
  }

  //Literal version of dft matrix.
  //LoadLiteralDFTMatrixToShared(inter_warp_id, buffer_RE, buffer_IM);

  //Load DFT matrix into the according fragments
  // wmma::load_matrix_sync(dft_RE_frag, buffer_RE, 16);
  // wmma::load_matrix_sync(dft_IM_frag, buffer_IM, 16);

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
  //sequential memory. 256 would be ideal -> possible improvement: fetch memory
  //for 8 matrices at once.
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
    //TODO: test speed and accuracy of cos,cosf,hcos
    //and literal version (look up table of N points cos(2*PI*i/N ))
    //Float because static_cast<__half>(combined_fft_length) overflows
    float tmp = static_cast<float>(i * j)  /
                static_cast<float>(combined_fft_length);
    // __half phase = __hmul(__hmul(2.0, static_cast<__half>(M_PI)),
    //                       static_cast<__half>(tmp));
    float phase = 2.0 * M_PI * tmp;

    //TO-SELF: test cosf vs cos accuracy and speed
    __half twiddle_RE = cosf(phase);
    __half twiddle_IM = -sinf(phase);

    //Fetch current data
    __half2 tmp_point = input_data[global_memory_offset];

    //Unpacking and sequential scaling
    __half input_RE = __hdiv(tmp_point.x, static_cast<__half>(16.0));
    __half input_IM = __hdiv(tmp_point.y, static_cast<__half>(16.0));

    //Store modified data to buffer arrays
    //mod_RE = RE*twid_RE - IM*twid_IM
    matrix_a_data_RE[buffer_matrix_memory_offset] =
        __hsub(__hmul(input_RE, twiddle_RE), __hmul(input_IM, twiddle_IM));
    //mod_IM = RE*twid_IM + IM*twid_RE
    matrix_a_data_IM[buffer_matrix_memory_offset] =
        __hfma(input_RE , twiddle_IM, __hmul(input_IM, twiddle_RE));
  }

  //Declare the fragments
  // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
  //     data_RE_frag;
  // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
  //     data_IM_frag;

  //Load the modified data from shared mem buffer
  // wmma::load_matrix_sync(data_RE_frag, buffer_RE, 16);
  // wmma::load_matrix_sync(data_IM_frag, buffer_IM, 16);

  // wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_1_frag;
  // wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_RE_2_frag;
  // wmma::fragment<wmma::accumulator, 16, 16, 16, half> accumulator_IM_frag;

  //Initialize the output to zero
  // wmma::fill_fragment(accumulator_RE_1_frag, 0.0f);
  // wmma::fill_fragment(accumulator_RE_2_frag, 0.0f);
  // wmma::fill_fragment(accumulator_IM_frag, 0.0);

  #pragma unroll
  for(int i=0; i<8; i++){
    matrix_acc_RE[i*inter_warp_id] = 0;
    matrix_acc_IM[i*inter_warp_id] = 0;
  }

  //Perform the matrix multiplication of two complex matrices AxB via 4 matrix
  //multiplications i.e. RE(AxB)=RE(A)xRE(B) - IM(A)xIM(B) and IM(AxB) =
  //RE(A)xIM(B) + IM(A)xRE(B)
  // wmma::mma_sync(accumulator_RE_1_frag, data_RE_frag, dft_RE_frag,
  //                accumulator_RE_1_frag);
  // wmma::mma_sync(accumulator_RE_2_frag, data_IM_frag, dft_IM_frag,
  //                accumulator_RE_2_frag);
  // wmma::mma_sync(accumulator_IM_frag, data_IM_frag, dft_RE_frag,
  //                accumulator_IM_frag);
  // wmma::mma_sync(accumulator_IM_frag, data_RE_frag, dft_IM_frag,
  //                accumulator_IM_frag);

  //wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 d, a, b, c;

  __half2* matrix_a_helper_RE = (__half2*)matrix_a_data_RE;
  __half2* matrix_a_helper_IM = (__half2*)matrix_a_data_IM;
  __half2* matrix_b_helper_RE = (__half2*)matrix_b_dft_RE;
  __half2* matrix_b_helper_IM = (__half2*)matrix_b_dft_IM;
  __half2* matrix_acc_helper_RE = (__half2*)matrix_acc_RE;
  __half2* matrix_acc_helper_IM = (__half2*)matrix_acc_IM;

  //a_IM*b_IM
  //doesnt work due to stupid restriction that thinks __half2 x in shared mem =/=
  //.reg .f16 x<2>  although it is the same :)
  //=> either use ptx for most / the entire kernel or see if it is possible to do
  //the modifications in the fragments
  asm ("wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27, %28, %29, %30, %31};" :
       "=f" (matrix_acc_helper_RE[0]), "=f" (matrix_acc_helper_RE[2]), "=f" (matrix_acc_helper_RE[4]), "=f" (matrix_acc_helper_RE[6]), "=f" (matrix_acc_helper_RE[8]), "=f" (matrix_acc_helper_RE[10]), "=f" (matrix_acc_helper_RE[12]), "=f" (matrix_acc_helper_RE[14]) :
       "f" (matrix_a_helper_IM[0]), "f" (matrix_a_helper_IM[2]), "f" (matrix_a_helper_IM[4]), "f" (matrix_a_helper_IM[6]), "f" (matrix_a_helper_IM[8]), "f" (matrix_a_helper_IM[10]), "f" (matrix_a_helper_IM[12]), "f" (matrix_a_helper_IM[14]),
       "f" (matrix_b_helper_IM[0]), "f" (matrix_b_helper_IM[2]), "f" (matrix_b_helper_IM[4]), "f" (matrix_b_helper_IM[6]), "f" (matrix_b_helper_IM[8]), "f" (matrix_b_helper_IM[10]), "f" (matrix_b_helper_IM[12]), "f" (matrix_b_helper_IM[14]),
       "f" (matrix_acc_helper_RE[0]), "f" (matrix_acc_helper_RE[2]), "f" (matrix_acc_helper_RE[4]), "f" (matrix_acc_helper_RE[6]), "f" (matrix_acc_helper_RE[8]), "f" (matrix_acc_helper_RE[10]), "f" (matrix_acc_helper_RE[12]), "f" (matrix_acc_helper_RE[14])
      );

  // //multiply by -1
  // #pragma unroll
  // for(int i=0; i<8; i++){
  //   matrix_acc_RE[i*inter_warp_id] = -matrix_acc_RE[i*inter_warp_id];
  // }

  // //a_RE * b_RE - (a_IM * b_IM) i.e. RE(AxB)
  // asm ("wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27, %28, %29, %30, %31};" :
  //      "r=" (*static_cast<__half2*>(&(matrix_acc_RE[0]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[2]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[4]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[6]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[8]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[10]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[12]))), "r=" (*static_cast<__half2*>(&(matrix_acc_RE[14]))) :
  //      "r" (*static_cast<__half2*>(&(matrix_a_data_RE[0]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[2]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[4]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[6]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[8]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[10]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[12]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[14]))),
  //      "r" (*static_cast<__half2*>(&(matrix_b_data_RE[0]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[2]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[4]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[6]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[8]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[10]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[12]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[14]))),
  //      "r" (*static_cast<__half2*>(&(matrix_acc_RE[0]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[2]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[4]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[6]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[8]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[10]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[12]))), "r" (*static_cast<__half2*>(&(matrix_acc_RE[14])))
  //     );
  //
  // //a_RE * b_IM
  // asm ("wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27, %28, %29, %30, %31};" :
  //      "r=" (*static_cast<__half2*>(&(matrix_acc_IM[0]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[2]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[4]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[6]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[8]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[10]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[12]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[14]))) :
  //      "r" (*static_cast<__half2*>(&(matrix_a_data_RE[0]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[2]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[4]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[6]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[8]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[10]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[12]))), "r" (*static_cast<__half2*>(&(matrix_a_data_RE[14]))),
  //      "r" (*static_cast<__half2*>(&(matrix_b_data_IM[0]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[2]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[4]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[6]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[8]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[10]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[12]))), "r" (*static_cast<__half2*>(&(matrix_b_data_IM[14]))),
  //      "r" (*static_cast<__half2*>(&(matrix_acc_IM[0]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[2]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[4]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[6]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[8]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[10]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[12]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[14])))
  //     );
  //
  // //a_IM * b_RE + a_RE * b_IM i.e. IM(AxB)
  // asm ("wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27, %28, %29, %30, %31};" :
  //      "r=" (*static_cast<__half2*>(&(matrix_acc_IM[0]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[2]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[4]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[6]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[8]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[10]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[12]))), "r=" (*static_cast<__half2*>(&(matrix_acc_IM[14]))) :
  //      "r" (*static_cast<__half2*>(&(matrix_a_data_IM[0]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[2]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[4]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[6]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[8]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[10]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[12]))), "r" (*static_cast<__half2*>(&(matrix_a_data_IM[14]))),
  //      "r" (*static_cast<__half2*>(&(matrix_b_data_RE[0]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[2]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[4]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[6]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[8]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[10]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[12]))), "r" (*static_cast<__half2*>(&(matrix_b_data_RE[14]))),
  //      "r" (*static_cast<__half2*>(&(matrix_acc_IM[0]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[2]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[4]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[6]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[8]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[10]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[12]))), "r" (*static_cast<__half2*>(&(matrix_acc_IM[14])))
  //     );

  //Store IM part of the output
  // wmma::store_matrix_sync(buffer_RE, accumulator_RE_1_frag, 16,
  //                         wmma::mem_row_major);
  // wmma::store_matrix_sync(buffer_tmp_RE, accumulator_RE_2_frag, 16,
  //                         wmma::mem_row_major);
  // wmma::store_matrix_sync(buffer_IM, accumulator_IM_frag, 16,
  //                         wmma::mem_row_major);

  //RE = RE_1 + RE_2, IM = IM_1 + IM_2
  // #pragma unroll
  // for(int k=0; k<8; k++){
  //   int buffer_array_id = inter_warp_id_16 +
  //                         (16 *  (k + (8 * inter_warp_id_is_upper_16)));
  //
  //   buffer_RE[buffer_array_id] -= buffer_tmp_RE[buffer_array_id];
  // }

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

    __half2 tmp = {matrix_acc_RE[buffer_matrix_memory_offset],
                   matrix_acc_IM[buffer_matrix_memory_offset]};

    output_data[global_memory_offset] = tmp;
  }
}
