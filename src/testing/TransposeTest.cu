//Example of usage of the FFT implementation with synthetic data
#include <iostream>
#include <string>
#include <optional>
#include <memory>

#include "../base/Plan.h"
#include "../base/ComputeFFT.h"
#include "WeightMaker.h"
#include "DataMaker.cu"
#include "FileWriter.h"

int main(){
  int fft_length = 16*16*16*16;

  //Get weights for data creation
  //int amount_of_frequencies = 10;
  int amount_of_frequencies = 5;
  std::unique_ptr<float2[]> weights =
      std::make_unique<float2[]>(amount_of_frequencies);
  //SetRandomWeights(weights.get(), amount_of_frequencies, 42*42);
  SetDummyWeightsRE1(weights.get());

  float2* dptr_weights = nullptr;
  cudaMalloc(&dptr_weights, sizeof(float2) * amount_of_frequencies);
  cudaMemcpy(dptr_weights, weights.get(),
             sizeof(float2) * amount_of_frequencies, cudaMemcpyHostToDevice);

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Allocate device memory
  __half2* dptr_input_data = nullptr;
  __half2* dptr_output_data = nullptr;
  cudaMalloc(&dptr_input_data, 2 * sizeof(__half2) * fft_length);
  dptr_output_data = dptr_input_data + fft_length;

  //Allocate mem on host for results
  std::unique_ptr<__half2[]> results1 = std::make_unique<__half2[]>(fft_length);
  //Allocate mem on host for results
  std::unique_ptr<__half2[]> results2 = std::make_unique<__half2[]>(fft_length);

  //Produce input data based on weights
  SineSupperposition<int,__half2><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_input_data, dptr_weights, amount_of_frequencies);

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  if (cudaMemcpy(results1.get(),
                 dptr_input_data,
                 fft_length * sizeof(__half2),
                 cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
     return false;
  }

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Write results to file
  WriteFFTToFile<__half2>("example_input.dat", fft_length, results1.get());

  Transposer<int><<<fft_length / 4096, 512, 32768>>>(
      dptr_input_data, dptr_output_data, fft_length, 3, 0);

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Copy results back
  if (cudaMemcpy(results2.get(),
                 dptr_output_data,
                 fft_length * sizeof(__half2),
                 cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
     return false;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  //Write results to file
  WriteFFTToFile<__half2>("example_trans.dat", fft_length, results2.get());

  TransposeKernel<<<fft_length / 512, 512>>>(
      dptr_input_data, dptr_output_data, fft_length, 3, 0);

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Copy results back
  if (cudaMemcpy(results2.get(),
                 dptr_output_data,
                 fft_length * sizeof(__half2),
                 cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
     return false;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  //Write results to file
  WriteFFTToFile<__half2>("example_trans_old.dat", fft_length, results2.get());

  for(int i=0; i<fft_length;i++){
    if (results1[i].x != results2[i].x && results1[i].y != results2[i].y) {
      std::cout << "Error at " << i << std::endl; 
    }
  }

  cudaFree(dptr_weights);
  cudaFree(dptr_input_data);

  return true;
}
