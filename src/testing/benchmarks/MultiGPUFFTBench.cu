//Used to benchmark the function ComputeFFTsMultiGPU
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
#include "../../base/ComputeFFTsMultiGPU.cu"
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
  int amount_of_GPUs = 4;

  std::vector<int> fft_length;
  std::vector<double> avg_runtime;
  std::vector<double> sigma_runtime;
  std::vector<double> avg_copying;
  std::vector<double> sigma_copying;

  int length = 16 * 8;
  for(int i=8; i<=log_length_max; i++){
    length = length * 2;
    fft_length.push_back(length);

    std::vector<double> runtime;
    std::vector<double> copying;

    std::vector<float> weights;
    weights.push_back(1.0);
    sttd::vector<std::vector<std::unique_ptr<__half[]>>> data;
    for(int j=0; j<amount_of_GPUs; j++){
      std::vector<std::unique_ptr<__half[]>> tmp;
      for(int i=0; i<amount_of_asynch_ffts; i++){
        tmp.push_back(CreateSineSuperpostion(fft_length.back(),  weights));
      }
      data.push_back(std::move(tmp));
    }


    std::vector<std::vector<Plan>> my_plan;
    for(int k=0; k<sample_size; k++){

      std::vector<std::vector<Plan> my_plan;
      for(int j=0; j<amount_of_GPUs; j++){
        std::vector<Plan> tmp_plan;
        for(int i=0; i<amount_of_asynch_ffts; i++){
          if (CreatePlan(fft_length.back()) {
            tmp_plan.push_back(CreatePlan(fft_length.back()));
          } else {
            std::cout << "Plan creation failed" << sttd::endl;
            return false;
          }
        }
        my_plan(std::move(tmp_plan));
      }

      std::vector<std::vector<DataHandler>> my_handler;
      for(int j=0; j<amount_of_GPUs; j++){
        cudaSetDevice(j);
        std::vector<DataHandler> tmp_handler;
        for(int i=0; i<amount_of_asynch_ffts; i++){
          tmp_handler.push_back(fft_length.back());
          error_mess = tmp_handler[i].PeakAtLastError().value_or("");
          if (error_mess != "") {
            std::cout << error_mess << std::endl;
            return false;
          }
        }
        my_handler.push_back(std::move(tmp_handler));
      }

      std::vector<std::vector>> streams;
      for(int j=0; j<amount_of_GPUs; j++){
        cudaSetDevice(j);
        std::vector<cudaStream_t> tmp_streams;
        tmp_streams.resize(amount_of_asynch_ffts);
        for(int i=0; i<amount_of_asynch_ffts; i++){
          if (cudaStreamCreate(&(ttmp_streams[i])) != cudaSuccess){
             std::cout << cudaGetErrorString(cudaPeekAtLastError())
                       << std::endl;
             return "Error while creating stream!";
          }
        }
        streams.push_back(std::move(tmp_streams));
      }

      std::vector<int> device_id_list;
      for(int j=0; j<amount_of_GPUs; j++){
        device_id_list.push_back(j);
        cudaSetDevice(j);
        for(int i=0; i<amount_of_asynch_ffts; i++){
          error_mess =
              my_handler[j][i].CopyDataHostToDeviceAsync(
                  data[j][i].get(), streams[j][i]).value_or("");
          if (error_mess != "") {
            std::cout << error_mess << std::endl;
            return false;
          }
        }
      }

      IntervallTimer computation_time;
      ComputeFFTsMultiGPU(device_id_list, my_plan, my_handler, streams);

      for(int j=0; j<amount_of_GPUs; j++){
        cudaSetDevice(j);
        cudaDeviceSynchronize();
      }

      double tmp = computation_time.getTimeInNanoseconds();
      runtime.push_back(tmp);

      for(int j=0; j<amount_of_GPUs; j++){
        cudaSetDevice(j);
        for(int i=0; i<amount_of_asynch_ffts; i++){
          error_mess = my_handler[j][i].CopyResultsDeviceToHostAsync(
              data[j][i].get(), my_plan[j][i].amount_of_r16_steps_,
              my_plan[j][i].amount_of_r2_steps_, streams[j][i]).value_or("");
          if (error_mess != "") {
            std::cout << error_mess << std::endl;
            return false;
          }
        }
      }

      for(int j=0; j<amount_of_GPUs; j++){
        cudaSetDevice(j);
        cudaDeviceSynchronize();
      }
    }
    avg_runtime.push_back(ComputeAverage(runtime));
    sigma_runtime.push_back(ComputeSigma(runtime, avg_runtime.back()));
  }

  WriteBenchResultsToFile(avg_runtime, sigma_runtime, fft_length,
                          std::to_string(sample_size));
  return true;
}
