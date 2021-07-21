//Used to benchmark the function ComputeFFTs
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

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
  int amount_of_asynch_ffts = 4;

  std::vector<int> fft_length;
  std::vector<double> avg_runtime;
  std::vector<double> sigma_runtime;

  int length = 16 * 8;
  for(int i=8; i<=log_length_max; i++){
    length = length * 2;
    fft_length.push_back(length);

    std::vector<double> runtime;

    std::vector<float> weights;
    weights.push_back(1.0);
    std::vector<std::unique_ptr<__half[]>> data;
    for(int i=0; i<amount_of_asynch_ffts; i++){
      data.push_back(CreateSineSuperpostion(fft_length.back(),  weights));
    }

    for(int k=0; k<sample_size; k++){
      std::vector<Plan> my_plan;
      for(int i=0; i<amount_of_asynch_ffts; i++){
        if (CreatePlan(fft_length.back()) {
          my_plan.push_back(CreatePlan(fft_length.back()));
        } else {
          std::cout << "Plan creation failed" << sttd::endl;
          return false;
        }
      }

      std::vector<DataHandler> my_handler;
      for(int i=0; i<amount_of_asynch_ffts; i++){
        my_handler.push_back(fft_length.back());
        error_mess = my_handler[i].PeakAtLastError().value_or("");
        if (error_mess != "") {
          std::cout << error_mess << std::endl;
          return false;
        }
      }

      std::vector<cudaStream_t> streams;
      streams.resize(amount_of_asynch_ffts);
      for(int i=0; i<amount_of_asynch_ffts; i++){
        if (cudaStreamCreate(&(streams[i])) != cudaSuccess){
           std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
           return "Error while creating stream!";
        }
      }

      for(int i=0; i<amount_of_asynch_ffts; i++){
        error_mess =
            my_handler[i].CopyDataHostToDeviceAsync(
                data[i].get(), streams[i]).value_or("");
        if (error_mess != "") {
          std::cout << error_mess << std::endl;
          return false;
        }
      }

      cudaDeviceSynchronize();
      IntervallTimer computation_time;

      error_mess = ComputeFFTs(my_plan, my_handler, streams).value_or("");
      if (error_mess != "") {
        std::cout << error_mess << std::endl;
        return false;
      }
      cudaDeviceSynchronize();
      runtime.push_back(computation_time.getTimeInNanoseconds());

      for(int i=0; i<amount_of_asynch_ffts; i++){
        error_mess = my_handler[i].CopyResultsDeviceToHostAsync(
            data[i].get(), my_plan[i].amount_of_r16_steps_,
            my_plan[i].amount_of_r2_steps_, streams[i]).value_or("");
        if (error_mess != "") {
          std::cout << error_mess << std::endl;
          return false;
        }
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
