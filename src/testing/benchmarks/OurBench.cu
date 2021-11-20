
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <optional>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../DataMaker.cu"
#include "../WeightMaker.h"
#include "../FileWriter.h"
#include "../../base/Plan.h"
#include "../../base/ComputeFFT.h"

struct BatchResult{
  double Average_;
  double RMS_;
  int fft_length_;
};

double GetAverage(std::vector<double> data){
  double tmp = 0;
  for(int i=0; i<static_cast<int>(data.size()); i++){
    tmp += data[i];
  }

  return (tmp/static_cast<double>(data.size()));
}

double GetRMS(std::vector<double> data, double average){
  double tmp = 0;

  for(int i=0; i<static_cast<int>(data.size()); i++){
    double tmp2 = data[i] - average;
    tmp += (tmp2 * tmp2);
  }

  return sqrt(tmp / static_cast<double>(data.size()));
}

int main(){
  constexpr int min_fft_length = 16*16*16;
  constexpr int max_fft_length = 16*16*16*16*16 * 16*16;
  constexpr int max_frequencies = 10;
  constexpr int samples = 100;
  constexpr int warmup_samples = 10;
  constexpr int total_samples = samples + warmup_samples;

  std::vector<BatchResult> results;

  int64_t current_fft_length = min_fft_length;

  while(current_fft_length <= max_fft_length){
    std::cout << "Current fft_length: " << current_fft_length << std::endl;

    std::unique_ptr<float2[]> weights =
        std::make_unique<float2[]>(max_frequencies);
    float2* dptr_weights = nullptr;
    cudaMalloc(&dptr_weights, sizeof(float2) * max_frequencies);

    SetRandomWeights(weights.get(), max_frequencies, 42*42);
    cudaMemcpy(dptr_weights, weights.get(),
               sizeof(float2) * max_frequencies,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    __half2 dptr_data;
    __half2 dptr_results;
    cudaMalloc((void**)(&dptr_data), static_cast<int>(sizeof(__half2) * current_fft_length));
    cudaMalloc((void**)(&dptr_results), static_cast<int>(sizeof(__half2) * current_fft_length));

    SineSupperposition<__half2><<<current_fft_length / 1024, 1024>>>(
        current_fft_length, dptr_data, dptr_weights, max_frequencies, 1.0);

    std::optional<Plan> possible_plan = MakePlan(current_fft_length);
    Plan my_plan;
    if (possible_plan) {
      my_plan = possible_plan.value();
    } else {
      std::cout << "Plan creation failed" << std::endl;
      return false;
    }

    std::vector<double> runtimes;

    for(int i=0; i<total_samples; i++){
      double runtime;

      SineSupperposition<__half2><<<current_fft_length / 1024, 1024>>>(
          current_fft_length, dptr_data, dptr_weights, max_frequencies, 1.0);

      cudaDeviceSynchronize();

      IntervallTimer timer;

      ComputeFFT<int>(my_plan, dptr_data, dptr_results);

      cudaDeviceSynchronize();

      runtime = static_cast<double>(timer.getTimeInNanoseconds());

      if(i >= warmup_samples) {
        runtimes.push_back(runtime);
      }
    }

    BatchResult current_result;
    current_result.Average_ = GetAverage(runtimes);
    current_result.RMS_ = GetRMS(runtimes, current_result.Average_);
    current_result.fft_length_ = current_fft_length;

    results.push_back(current_result);

    cudaFree(dptr_results);
    cudaFree(dptr_data);

    current_fft_length = current_fft_length * 16;
  }

  std::cout << "Finished Benchmark" << std::endl;

  WriteBenchResultsToFIle("BenchOur.dat", results);

  return true;
}
