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
  int log_length_max = 20;
  int sample_size = 10;

  std::vector<int> fft_length;
  std::vector<double> avg_runtime;
  std::vector<double> sigma_runtime;

  int length = 16 * 8;
  for(int i=8; i<=log_length_max; i++){
    length = length * 2;
    fft_length.push_back(length);

    std::vector<float> weights;
    weights.push_back(1.0);
    std::unique_ptr<__half[]> data =
        CreateSineSuperpostion(fft_length.back(),  weights);

    std::vector<double> runtime;

    for(int k=0; k<sample_size; k++){
      Plan my_plan;
      if (CreatePlan(fft_length.back())) {
        my_plan = CreatePlan(fft_length.back());
      } else {
        std::cout << "Plan creation failed" << sttd::endl;
        return false;
      }

      DataHandler my_handler(fft_length.back());
      error_mess = my_handler.PeakAtLastError().value_or("");
      if (error_mess != "") {
        std::cout << error_mess << std::endl;
        return false;
      }

      error_mess = my_handler.CopyDataHostToDevice(data.get()).value_or("");
      if (error_mess != "") {
        std::cout << error_mess << std::endl;
        return false;
      }

      cudaDeviceSynchronize();
      IntervallTimer computation_time;
      error_mess = ComputeFFT(my_plan, my_handler).value_or("");
      if (error_mess != "") {
        std::cout << error_mess << std::endl;
        return false;
      }

      cudaDeviceSynchronize();
      runtime.push_back(computation_time.getTimeInNanoseconds());

      error_mess = my_handler.CopyResultsDeviceToHost(
          data.get(), my_plan.amount_of_r16_steps_,
          my_plan.amount_of_r2_steps_).value_or("");
      if (error_mess != "") {
        std::cout << error_mess << std::endl;
        return false;
      }
      cudaDeviceSynchronize();
    }

    avg_runtime.push_back(ComputeAverage(runtime));
    sigma_runtime.push_back(ComputeSigma(runtime, avg_runtime.back()));
  }

  WriteBenchResultsToFile(avg_runtime, sigma_runtime, fft_length,
                          std::to_string(sample_size));
  return true;
}
