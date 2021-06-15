//This precompute batches of the dft matrix that is used in each call of the
//DFTKernel() and Radix16Kernel().
//The number of threads has to be equal to the amount of entries to be computed.
//I.e. for one batch exactly 16*16*16 threads have to be launched or e.g. for 4
//-> 4*16*16*16 threads.
__global__ void ComputeDFTMatrix(__half* dft_matrix_batch_RE,
                                 __half* dft_matrix_batch_IM) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int inter_matrix_id = thread_id % (16 * 16);
  int row_id = inter_matrix_id % 16;
  int collum_id = inter_matrix_id / 16;

  float phase = (2 * row_id * collum_id * M_PI) / 16.0;
  dft_matrix_batch_RE[thread_id] = __float2half(cosf(phase));
  dft_matrix_batch_IM[thread_id] = __float2half(-sinf(phase));
}
