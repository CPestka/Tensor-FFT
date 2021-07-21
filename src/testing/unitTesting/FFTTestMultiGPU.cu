#pragma once

//Used to test functonality
#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../../base/Plan.cpp"
#include "../../base/ComputeFFT.cu"
#include "../../base/ComputeFFTMultiGPU.cu"
#include "../TestingDataCreation.cu"
#include "../FileWriter.cu"
#include "../AccuracyCalculator.h"
#include "CuFFTTest.cu"

bool TestMultiGPU(int fft_length, int amount_of_asynch_ffts,
                  int amount_of_GPUs, double avg_deviation_threshold,
                  double sigma_deviation_threshold){
  std::string file_name_prefix = "test_multi_GPU_" + std::to_string(fft_length);
  std::string comparison_data_file_name =
      ("test_comparison_" + std::to_string(fft_length)) + ".dat";

  std::optional<std::string> error_mess;

  error_mess = CreateComparisonData(fft_length, comparison_data_file_name);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Prepare input data on cpu
  std::vector<float> weights;
  weights.push_back(1.0);
  std::vector<std::vector<std::unique_ptr<__half[]>>> data;
  for(int j=0; j<amount_of_GPUs; j++){
    std::vector<std::unique_ptr<__half[]>> tmp;
    for(int i=0; i<amount_of_asynch_ffts; i++){
      tmp.push_back(CreateSineSuperpostion(fft_length, weights));
    }
    data.push_back(std::move(tmp));
  }


  //Get plan
  std::vector<std::vector<Plan>> my_plans;
  for(int j=0; j<amount_of_GPUs; j++){
    std::vector<Plan> tmp;
    for(int i=0; i<amount_of_asynch_ffts; i++){
      if (CreatePlan(fft_length)) {
        tmp.push_back(CreatePlan(fft_length).value());
      } else {
        std::cout << "Plan creation failed" << std::endl;
        return false;
      }
    }
    my_plans.push_back(std::move(tmp));
  }

  error_mess = WriteResultsREToFile(file_name_prefix + "_input.dat", fft_length,
                                    data[0][0].get());
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Construct a DataHandler for data on GPU
  std::vector<std::vector<DataHandler>> my_handlers;
  for(int j=0; j<amount_of_GPUs; j++){
    cudaSetDevice(j);
    std::vector<DataHandler> tmp;
    for(int i=0; i<amount_of_asynch_ffts; i++){
      tmp.push_back(fft_length);

      error_mess = tmp.PeakAtLastError();
      if (error_mess) {
        std::cout << error_mess.value() << std::endl;
        return false;
      }
    }
    my_handlers.push_back(std::move(tmp));
  }


  //Create a stream for each fft
  std::vector<std::vector<cudaStream_t>> streams;
  for(int j=0; j<amount_of_GPUs; j++){
    cudaSetDevice(j);
    std::vector<cudaStream_tr> tmp;
    tmp.resize(amount_of_asynch_ffts);
    for(int i=0; i<amount_of_asynch_ffts; i++){
      if (cudaStreamCreate(&(tmp[i])) != cudaSuccess){
         std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
         return false;
      }
    }
    streams.push_back(std::move(tmp));
  }


  //Copy data to gpu
  std::vector<int> device_id_list;
  for(int j=0; j<amount_of_GPUs; j++){
    device_id_list.push_back(j);
    cudaSetDevice(j);

    for(int i=0; i<amount_of_asynch_ffts; i++){
      error_mess = my_handlers[j][i].CopyDataHostToDeviceAsync(data[j][i].get(),
                                                               streams[j][i]);
      if (error_mess) {
        std::cout << error_mess.value() << std::endl;
        return false;
      }
    }
  }


  //Compute FFT
  ComputeFFTsMultiGPU(device_id_list, my_plans, my_handlers, streams);

  for(int j=0; j<amount_of_GPUs; j++){
    cudaSetDevice(j);

    for(int i=0; i<amount_of_asynch_ffts; i++){
      error_mess = my_handlers[j][i].CopyResultsDeviceToHostAsync(
          data[j][i].get(), my_plans[j][i].amount_of_r16_steps_,
          my_plans[j][i].amount_of_r2_steps_,streams[j][i]);
      if (error_mess) {
        std::cout << error_mess.value() << std::endl;
        return false;
      }
    }
  }

  for(int j=0; j<amount_of_GPUs; j++){
    cudaSetDevice(j);
    cudaDeviceSynchronize();
  }

  std::vector<std::string> file_names;
  for(int j=0; j<amount_of_GPUs; j++){
    for(int i=0; i<amount_of_asynch_ffts; i++){
      file_names.push_back(((((file_name_prefix + "_") + std::to_string(j))
                             + "_") + std::to_string(i)) + "_results.dat");
      error_mess = WriteResultsToFile(file_names.back(), fft_length,
                                      data[j][i].get());
      if (error_mess) {
        std::cout << error_mess.value() << std::endl;
        return false;
      }
    }
  }

  double avg = ComputeAverageDeviation(file_names, comparison_data_file_name);
  double sigma = ComputeSigmaOfDeviation(file_names,
                                         comparison_data_file_name, avg);
  if ((avg > avg_deviation_threshold) || (sigma > sigma_deviation_threshold)){
    std::cout << "Accuracy test failed!" << std::endl
              << "Avg: " << avg << " Threshold: " << avg_deviation_threshold
              << " Sigma: " << sigma << " Threshold: "
              << sigma_deviation_threshold << std::endl;
    return false;
  }

  return true;
}
