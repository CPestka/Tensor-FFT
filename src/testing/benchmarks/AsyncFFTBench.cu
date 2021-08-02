//Used to benchmark the function ComputeFFTs
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
  int log_length_max = 20;
  int sample_size = 25;
  int warmup_samples = 5;
  int amount_of_asynch_ffts = 20;

  std::vector<int> fft_length;
  std::vector<double> avg_runtime;
  std::vector<double> sigma_runtime;

  std::optional<std::string> error_mess;

  int length = 16 * 8;
  for(int i=8; i<=log_length_max; i++){
    length = length * 2;
    fft_length.push_back(length);
    std::cout << "Starting fft length: " << length << std::endl;

    std::vector<double> runtime;

    std::vector<float> weights;
    weights.push_back(1.0);
    std::unique_ptr<__half[]> data;
    data = CreateSineSuperpostionBatch(fft_length.back(),
                                       amount_of_asynch_ffts, weights);

    Plan my_plan;
    if (CreatePlan(fft_length.back())) {
      my_plan = CreatePlan(fft_length.back()).value();
    } else {
      std::cout << "Plan creation failed" << std::endl;
      return false;
    }
    /*
    if (CreatePlan(fft_length.back(), "../TunerResults.dat")) {
      my_plan = CreatePlan(fft_length.back(), "../TunerResults.dat").value();
    } else {
      std::cout << "Plan creation failed" << std::endl;
      return false;
    }
    */

    DataBatchHandler my_handler(fft_length.back(), amount_of_asynch_ffts);
    error_mess = my_handler.PeakAtLastError();
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    for(int k=0; k<sample_size + warmup_samples; k++){
      my_handler.CopyDataHostToDevice(data.get());

      cudaDeviceSynchronize();

      IntervallTimer computation_time;

      error_mess = ComputeFFTs(my_plan, my_handler);
      if (error_mess) {
        std::cout << error_mess.value() << std::endl;
        return false;
      }

      cudaDeviceSynchronize();

      if (k >= warmup_samples) {
        runtime.push_back(computation_time.getTimeInNanoseconds());
      }
    }

    avg_runtime.push_back(ComputeAverage(runtime));
    sigma_runtime.push_back(ComputeSigma(runtime, avg_runtime.back()));
  }

  WriteBenchResultsToFile(avg_runtime, sigma_runtime, fft_length,
                          (std::to_string(sample_size) + "_async_") +
                          std::to_string(amount_of_asynch_ffts));
  return true;
}
