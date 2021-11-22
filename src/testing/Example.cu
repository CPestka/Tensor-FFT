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
  int fft_length = 16*16*16;

  std::optional<Plan> possible_plan = MakePlan(fft_length);
  Plan my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  std::cout << my_plan.amount_of_r16_steps_ << "\n"
            << my_plan.amount_of_r2_steps_ << "\n"
            << my_plan.amount_of_r16_kernels_ << "\n"
            << my_plan.results_in_results_ << "\n"
            << my_plan.transpose_config_.warps_per_block_ << "\n"
            << my_plan.transpose_config_.blocksize_ << "\n"
            << my_plan.transpose_config_.gridsize_ << "\n"
            << my_plan.transpose_config_.shared_mem_in_bytes_ << "\n"
            << my_plan.base_fft_config_.warps_per_block_ << "\n"
            << my_plan.base_fft_config_.blocksize_ << "\n"
            << my_plan.base_fft_config_.gridsize_ << "\n"
            << my_plan.base_fft_config_.shared_mem_in_bytes_ << "\n"
            << my_plan.r16_config_.warps_per_block_ << "\n"
            << my_plan.r16_config_.blocksize_ << "\n"
            << my_plan.r16_config_.gridsize_ << "\n"
            << my_plan.r16_config_.shared_mem_in_bytes_ << "\n";

  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    std::cout << "Error! Plan imcompatible with used device." << std::endl;
    return false;
  }

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
  std::unique_ptr<__half2[]> results = std::make_unique<__half2[]>(fft_length);

  //Produce input data based on weights
  SineSupperposition<int,__half2><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_input_data, dptr_weights, amount_of_frequencies);

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Copy results back
  if (cudaMemcpy(results.get(),
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
  WriteFFTToFile<__half2>("example_input.dat", fft_length, results.get());

  //Compute the FFT on the device
  std::optional<std::string> error_mess =
      ComputeFFT<int>(my_plan, dptr_input_data, dptr_output_data,
                      GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Copy results back
  if (cudaMemcpy(results.get(),
                 my_plan.results_in_results_ ?
                     dptr_output_data : dptr_input_data,
                 fft_length * sizeof(__half2),
                 cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
     return false;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  //Write results to file
  WriteFFTToFile<__half2>("example_results.dat", fft_length, results.get());

  std::unique_ptr<__half2[]> comp_data =
      GetComparisionFP16Data(dptr_weights, amount_of_frequencies, fft_length,
                             1.0);

  std::cout << "beep" << std::endl;
  std::unique_ptr<cufftDoubleComplex[]> comp_data1 =
      GetComparisionFP64Data(dptr_weights, amount_of_frequencies, fft_length,
                             1.0);
  std::cout << "beep" << std::endl;
  //Write results to file
  WriteFFTToFile<__half2>("example_cu_results.dat", fft_length,
                          comp_data.get());

  cudaFree(dptr_weights);
  cudaFree(dptr_input_data);

  return true;
}
