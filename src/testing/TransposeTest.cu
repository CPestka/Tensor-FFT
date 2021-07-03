//Used to test correctness of transposer
#include <iostream>
#include <cstdint>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../base/Plan.cpp"
#include "../base/ComputeFFT.cu"
#include "TestingDataCreation.cu"
#include "../base/Timer.h"
#include "FileWriter.cu"

int main(){
  int fft_length = 16*16*16;
  int amount_of_r16_steps = 2;
  int amount_of_r2_steps = 0;
  int transpose_blocksize = 256;

  std::vector<float> weights;
  weights.push_back(1.0);
  //weights.push_back(1.4);
  std::unique_ptr<__half[]> data = CreateSineSuperpostion(fft_length, weights);

  WriteResultsToFile("input.dat", fft_length, data.get());

  std::unique_ptr<__half[]> transposed_data =
      std::make_unique<__half[]>(2 * fft_length);

  for(int i=0; i<fft_length*2; i++){
    transposed_data[i] = data[i];
  }

  __half tmp_RE[16][16][16];
  __half tmp_IM[16][16][16];
  for(int i=0; i<16; i++){
    for(int j=0; j<16; j++){
      for(int k=0; k<16; k++){
        tmp_RE[k][j][i] = transposed_data[i + 16*j + 16*16*k];
        tmp_RE[k][j][i] = transposed_data[i + 16*j + 16*16*k + fft_length];
      }
    }
  }
  __half tmp1_RE[16][16][16];
  __half tmp1_IM[16][16][16];

  //do the transposes i.e. from x[i][j][k] to x[k][j][i]
  for(int i=0; i<16; i++){
    for(int j=0; j<16; j++){
      for(int k=0; k<16; k++){
        tmp1_RE[k][j][i] = tmp_RE[i][k][j];
        tmp1_IM[k][j][i] = tmp_IM[i][k][j];
        transposed_data[i + 16*j + 16*16*k] = tmp1_RE[k][j][i];
        transposed_data[i + 16*j + 16*16*k + fft_length] = tmp1_IM[k][j][i];
      }
    }
  }

  WriteResultsToFile("transposed_test.dat", fft_length, transposed_data.get());

  //Allocation of in/out data arrays on the device
  __half* dptr_input_RE;
  __half* dptr_input_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  cudaMalloc((void**)(&dptr_input_RE), 4 * sizeof(__half) * fft_length);

  dptr_input_IM = dptr_input_RE + fft_length;
  dptr_results_RE = dptr_input_IM + fft_length;
  dptr_results_IM = dptr_results_RE + fft_length;

  //Memcpy of input data to device
  cudaMemcpy(dptr_input_RE, data, 2 * fft_length * sizeof(__half),
                 cudaMemcpyHostToDevice);

  int amount_of_transpose_blocks =
     ceil(static_cast<float>(fft_length) /
          static_cast<float>(transpose_blocksize));
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  TransposeKernel<<<amount_of_transpose_blocks, transpose_blocksize>>>(
      dptr_input_RE, dptr_input_IM, dptr_results_RE, dptr_results_IM,
      fft_length, amount_of_r16_steps, amount_of_r2_steps);

  //Memcpy of input data to device
  cudaMemcpy(data, dptr_input_RE, 2 * fft_length * sizeof(__half),
                 cudaMemcpyDeviceToHost);

  WriteResultsToFile("transposed_kernel.dat", fft_length, data.get());
}
