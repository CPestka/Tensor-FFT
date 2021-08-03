//Used to benchmark the function ComputeFFT
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../TestingDataCreation.cu"
#include "../FileWriter.cu"
#include "../Timer.h"
#include "../../base/ComputeFFT.cu"
#include "../../base/Plan.cpp"

double ComputeAverage(std::vector<double> data){
  double tmp = 0;
  for(int i=0; i<static_cast<int>(data.size()); i++){
    tmp += data[i];
  }
  return (tmp / (static_cast<double>(data.size()) - 1));
}

double ComputeSigma(std::vector<double> data, double average){
  double tmp = 0;
  for(int i=0; i<static_cast<int>(data.size()); i++){
    double tmp_1 = data[i] - average;
    tmp += (tmp_1 * tmp_1);
  }
  return sqrt(tmp / (static_cast<double>(data.size()) - 1));
}

int main(){
  int log_length_max = 12;
  int sample_size = 20;
  int warmup_samples = 5;

  std::vector<long long> fft_length;
  std::vector<double> avg_runtime;
  std::vector<double> sigma_runtime;

  std::optional<std::string> error_mess;

  int length = 16 * 8;
  for(int i=8; i<=log_length_max; i++){
    length = length * 2;
    fft_length.push_back(length);
    std::cout << "Starting fft length: " << length << std::endl;

    std::vector<float> weights;
    weights.push_back(1.0);
    std::unique_ptr<__half2[]> data =
        CreateSineSuperpostionH2(fft_length.back(),  weights);

    std::vector<double> runtime;

    __half2* dptr_data;
    __half2* dptr_results;
    cudaMalloc(&dptr_data, sizeof(__half2) * fft_length.back());
    cudaMalloc(&dptr_results, sizeof(__half2) * fft_length.back());

    cufftHandle plan;
    cufftResult r;

    r = cufftCreate(&plan);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan creation failed." << std::endl;
      return false;
    }

    size_t size = 0;
    r = cufftXtMakePlanMany(plan, 1, &fft_length.back(), nullptr, 1, 1,
                            CUDA_C_16F, nullptr, 1, 1, CUDA_C_16F, 1, &size,
                            CUDA_C_16F);
    if (r != CUFFT_SUCCESS) {
      std::cout << "Error! Plan creation failed." << std::endl;
      return false;
    }

    for(int k=0; k<sample_size + warmup_samples; k++){
      cudaMemcpy(dptr_data, data.get(), fft_length.back() * sizeof(__half2),
                 cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      IntervallTimer computation_time;

      r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
      if (r != CUFFT_SUCCESS) {
        std::cout << "Error! Plan execution failed." << std::endl;
        return false;
      }

      cudaDeviceSynchronize();

      if (k >= warmup_samples) {
        runtime.push_back(computation_time.getTimeInNanoseconds());
      }
    }

    avg_runtime.push_back(ComputeAverage(runtime));
    sigma_runtime.push_back(ComputeSigma(runtime, avg_runtime.back()));

    cufftDestroy(plan);
    cudaFree(dptr_results);
    cudaFree(dptr_data);
  }

  WriteBenchResultsToFile(avg_runtime, sigma_runtime, fft_length,
                          std::to_string(sample_size) + "_cuFFT");
  return true;
}
