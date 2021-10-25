//Example of usage of the FFT implementation with synthetic data
#include <iostream>
#include <string>
#include <optional>
#include <memory>

#include "../base/Plan.h"
#include "../base/ComputeFFT.h"
#include "WeightMaker.h"
#include "DataMaker.h"
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
  
  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    std::cout << "Error! Plan imcompatible with used device." << std::endl;
    return false;
  }

  //Get weights for data creation
  int amount_of_frequencies = 10;
  std::unique_ptr<float2[]> weights =
      std::make_unique<float2[]>(amount_of_frequencies);
  SetRandomWeights(weights.get(), amount_of_frequencies, 42*42);
  float2* dptr_weigths = nulölptr;
  cudaMalloc(&dptr_weigths, sizeof(float2) * amount_of_frequencies);
  cudaMemcpy(dptr_weigths, weights.get(),
             sizeof(float2) * amount_of_frequencies, cudaMemcpyHostToDevice);

  //Allocate device memory
  __half2* dptr_input_data = nullptr;
  __half2* dptr_output_data = nullptr;
  cudaMalloc(&dptr_input_data, 2 * sizeof(__half2) * fft_length);
  dptr_output_data = dptr_input_data + fft_length;

  //Produce input data based on weights
  SineSupperposition<__half2><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_input_data, dptr_weights, amount_of_frequencies);

  //Compute the FFT on the device
  std::optional<std::string> error_mess =
      ComputeFFT(my_plan, dptr_input_data, dptr_output_data,
                 GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Needed if data set smaller than 64KB and can be removed otherwise.
  cudaDeviceSynchronize();

  //Allocate mem on host for results
  std::unique_ptr<__half2[]> results = std::make_unique<__half2[]>(fft_length);

  //Copy results back
  if (cudaMemcpy(results.get(),
                 my_plan.results_in_results_ ?
                     dptr_output_data : dptr_input_data,
                 fft_length_ * sizeof(__half2),
                 cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
     return false;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  //Write results to file
  WriteFFTToFile("example_results.dat", fft_length, results.get());

  return true;
}
