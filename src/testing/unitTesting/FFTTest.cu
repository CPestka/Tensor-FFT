#pragma once

//Used to test functonality
#include <iostream>
#include <string>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../../base/Plan.cpp"
#include "../../base/DataHandler.cu"
#include "../../base/ComputeFFT.cu"
#include "../TestingDataCreation.cu"
#include "../FileWriter.cu"
#include "../AccuracyCalculator.h"
#include "CuFFTTest.cu"

std::optional<std::string> FullSingleFFTComputation(int fft_length,
                                                    std::string file_name){
  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data = CreateSineSuperpostion(fft_length,  weights);

  Plan my_plan;
  if (CreatePlan(fft_length)) {
    my_plan = CreatePlan(fft_length).value();
  } else {
    return "Plan creation failed";
  }

  std::optional<std::string> error_mess;

  error_mess = WriteResultsREToFile("input" + file_name, fft_length,
                                    data.get());
  if (error_mess) {
    return error_mess;
  }

  //Construct a DataHandler for data on GPU
  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    return error_mess;
  }

  //Copy data to gpu
  error_mess = my_handler.CopyDataHostToDevice(data.get());
  if (error_mess) {
    return error_mess;
  }

  //Compute FFT
  error_mess = ComputeFFT(my_plan, my_handler);
  if (error_mess) {
    return error_mess;
  }

  //Copy results back to cpu
  error_mess = my_handler.CopyResultsDeviceToHost(
      data.get(), my_plan.amount_of_r16_steps_,
      my_plan.amount_of_r2_steps_);
  if (error_mess) {
    return error_mess;
  }

  cudaDeviceSynchronize();

  error_mess = WriteResultsToFile(file_name, fft_length, data.get());
  if (error_mess) {
    return error_mess;
  }

  return std::nullopt;
}

std::optional<std::string> FullAsyncFFTComputation(
    int fft_length, int amount_of_asynch_ffts,
    std::vector<std::string> file_name){
  //Prepare input data on cpu
  std::vector<float> weights;
  weights.push_back(1.0);
  std::vector<std::unique_ptr<__half[]>> data;
  for(int i=0; i<amount_of_asynch_ffts; i++){
    data.push_back(CreateSineSuperpostion(fft_length, weights));
  }

  //Get plan
  std::vector<Plan> my_plans;
  for(int i=0; i<amount_of_asynch_ffts; i++){
    if (CreatePlan(fft_length)) {
      my_plans.push_back(CreatePlan(fft_length).value());
    } else {
      return "Plan creation failed";
    }
  }

  std::string error_mess;

  error_mess = WriteResultsToFile("input" + file_name[0], fft_length,
                                  data.get());
  if (error_mess != "") {
    return error_mess;
  }

  //Construct a DataHandler for data on GPU
  std::vector<DataHandler> my_handlers;
  for(int i=0; i<amount_of_asynch_ffts; i++){
    my_handlers.push_back(fft_length);

    error_mess = my_handlers[i].PeakAtLastError().value_or("");
    if (error_mess != "") {
      return error_mess;
    }
  }

  //Create a stream for each fft
  std::vector<cudaStream_t> streams;
  streams.resize(amount_of_asynch_ffts);
  for(int i=0; i<amount_of_asynch_ffts; i++){
    if (cudaStreamCreate(&(streams[i])) != cudaSuccess){
       std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
       return "Error while creating stream!";
    }
  }

  //Copy data to gpu
  for(int i=0; i<amount_of_asynch_ffts; i++){
    error_mess = my_handlers[i].CopyDataHostToDeviceAsync(
        data[i].get(), streams[i]).value_or("");
    if (error_mess != "") {
      return error_mess;
    }
  }


  //Compute FFT
  error_mess = ComputeFFTs(my_plans, my_handlers, streams).value_or("");
  if (error_mess != "") {
    return error_mess;
  }

  //Copy results back to cpu
  for(int i=0; i<amount_of_asynch_ffts; i++){
    eerror_mess = my_handlers[i].CopyResultsDeviceToHostAsync(
        data[i].get(), my_plans[i].amount_of_r16_steps_,
        my_plans[i].amount_of_r2_steps_, streams[i]).value_or("");
    if (error_mess != "") {
      return error_mess;
    }
  }

  cudaDeviceSynchronize();

  for(int i=0; i<amount_of_asynch_ffts; i++){
    error_mess = WriteResultsToFile(file_name[i], fft_length, data[i].get());
    if (error_mess != "") {
      return error_mess;
    }
  }

  return std::nullopt;
}

bool TestFullFFT(int fft_length,
                 double avg_deviation_threshold,
                 double sigma_deviation_threshold){
  std::optional<std::string>> err;

  std::string comparison_data_file_name =
    ("test_comparison_" + std::to_string(fft_length)) + ".dat";
  std::string data_file_name =
    ("test_" + std::to_string(fft_length)) + ".dat";
  err = CreateComparisionData(fft_length, comparison_data_file_name);
  if (err) {
    std::cout << err << std::endl;
    return false;
  }

  err = FullSingleFFTComputation(fft_length, data_file_name);
  if (err) {
    std::cout << err << std::endl;
    return false;
  }

  double avg = ComputeAverageDeviation(comparison_data_file_name,
                                       data_file_name);
  double sigma = ComputeSigmaOfDeviation(comparison_data_file_name,
                                         data_file_name, avg);
  if ((avg > avg_deviation_threshold) || (sigma > sigma_deviation_threshold)){
    std::cout << "Accuracy test failed!" << std::endl
              << "Avg: " << avg << " Threshold: " << avg_deviation_threshold
              << " Sigma: " << sigma << " Threshold: "
              << sigma_deviation_threshold << std::endl;
    return false;
  }

  return true;
}

bool TestFullFFTAsynch(int fft_length, int amount_of_asynch_ffts,
                       double avg_deviation_threshold,
                       double sigma_deviation_threshold){
std::optional<std::string>> err;

std::string comparison_data_file_name =
  ("test_comparison_" + std::to_string(fft_length)) + ".dat";
std::vector<std::string> data_file_name;
for(int i=0; i<amount_of_asynch_ffts; i++){
  data_file_name.push_back(((("test_" + std::to_string(fft_length)) + "_async_")
                            + std::to_string(i)) + ".dat");
}

err = CreateComparisionData(fft_length, comparison_data_file_name);
if (err) {
  std::cout << err << std::endl;
  return false;
}

err = FullAsyncFFTComputation(fft_length, amount_of_asynch_ffts,
                              data_file_name);
if (err) {
  std::cout << err << std::endl;
  return false;
}

double avg = ComputeAverageDeviation(data_file_name, comparison_data_file_name);
double sigma = ComputeSigmaOfDeviation(data_file_name,
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
