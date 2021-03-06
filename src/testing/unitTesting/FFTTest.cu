#pragma once

//Used to test functonal correctness of results by comparing our fft results
//to cuffts results using and accuracy cutoff to determine success.
#include <iostream>
#include <string>
#include <memory>
#include <optional>
#include <vector>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../../base/Plan.h"
#include "../../base/DataHandler.h"
#include "../../base/ComputeFFT.h"
#include "../TestingDataCreation.h"
#include "../FileWriter.h"
#include "../AccuracyCalculator.h"
#include "CuFFTTest.h"

template <typename Integer>
std::optional<std::unique_ptr<__half[]>> FullSingleFFTComputation(
    const Integer fft_length,
    const std::vector<float> weights_RE,
    const std::vector<float> weights_IM,
    const Integer frequency_cutof){

  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionHGPU(fft_length, weights_RE, weights_IM,
                                 frequency_cutof);

  std::optional<Plan<Integer>> possible_plan = CreatePlan(fft_length);
  Plan<Integer> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return std::nullopt;
  }

  std::optional<std::string> error_mess;

  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    std::cout << "Error Plan doesnt work on used device." << std::endl;
    return std::nullopt;
  };

  //Construct a DataHandler for data on GPU
  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return std::nullopt;
  }

  //Copy data to gpu
  error_mess = my_handler.CopyDataHostToDevice(data.get());
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return std::nullopt;
  }

  //Compute FFT
  error_mess = ComputeFFT(my_plan, my_handler,
                          GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return std::nullopt;
  }

  //Copy results back to cpu
  error_mess = my_handler.CopyResultsDeviceToHost(
      data.get(), my_plan.results_in_results_);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return std::nullopt;
  }

  cudaDeviceSynchronize();

  return std::move(data);
}

template <typename Integer>
std::optional<std::string> FullSingleFFTComputation(
    const Integer fft_length,
    const std::string file_name){
  std::vector<float> weights_RE{ 0.0, 0.0 };
  std::vector<float> weights_IM{ 1.0, 0.0 };
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionHGPU(fft_length, weights_RE, weights_IM);

  std::optional<Plan<Integer>> possible_plan = CreatePlan(fft_length);
  Plan<Integer> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return std::nullopt;
  }

  std::optional<std::string> error_mess;

  error_mess = WriteResultsToFile(("test_input_" + std::to_string(fft_length))
                                   + ".dat", fft_length, data.get());
  if (error_mess) {
    return error_mess;
  }

  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    return "Error Plan doesnt work on used device.";
  };

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
  error_mess = ComputeFFT(my_plan, my_handler,
                          GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    return error_mess;
  }

  //Copy results back to cpu
  error_mess = my_handler.CopyResultsDeviceToHost(
      data.get(), my_plan.results_in_results_);
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


template <typename Integer>
std::optional<std::string> FullAsyncFFTComputation(
    const Integer fft_length,
    const int amount_of_asynch_ffts,
    const std::vector<std::string> file_name){
  //Prepare input data on cpu
  std::vector<float> weights_RE{ 0.0, 0.0 };
  std::vector<float> weights_IM{ 1.0, 0.0 };
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionBatch(fft_length, amount_of_asynch_ffts,
                                  weights_RE, weights_IM);

  std::optional<Plan<Integer>> possible_plan = CreatePlan(fft_length);
  Plan<Integer> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return std::nullopt;
  }

  std::optional<std::string> error_mess;

  for(int i=0; i<amount_of_asynch_ffts; i++){
    error_mess = WriteResultsToFile(((("test_async_input_" +
                                       std::to_string(fft_length))
                                       + "_") + std::to_string(i)) + ".dat",
                                       fft_length,
                                       data.get() + (i * 2 * fft_length));
    if (error_mess) {
      return error_mess;
    }
  }

  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  if (!PlanWorksOnDevice(my_plan, device_id)) {
    return "Error Plan doesnt work on used device.";
  };

  //Construct a DataHandler for data on GPU
  DataBatchHandler my_handler(fft_length, amount_of_asynch_ffts);
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
  error_mess = ComputeFFT(my_plan, my_handler,
                          GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    return error_mess;
  }

  //Copy results back to cpu
  error_mess = my_handler.CopyResultsDeviceToHost(
      data.get(), my_plan.results_in_results_);
  if (error_mess) {
    return error_mess;
  }

  cudaDeviceSynchronize();

  error_mess = WriteResultBatchToFile(file_name, fft_length, data.get());
  if (error_mess) {
    return error_mess;
  }


  return std::nullopt;
}

template <typename Integer>
bool TestFullFFT(const Integer fft_length,
                 const double avg_deviation_threshold,
                 const double sigma_deviation_threshold,
                 const double max_deviation_threshold,
                 const std::vector<float> weights_RE,
                 const std::vector<float> weights_IM){
  std::optional<std::string> err;

  //Compute comparision data and check validity
  auto possible_comparission_data =
      CreateComparisonDataDouble(fft_length, weights_RE, weights_IM);
  if (!possible_comparission_data) {
    std::cout << "Error! Failed to create comparision data." << std::endl;
    return false;
  }
  std::unique_ptr<double[]> comparission_data =
      ConvertResultsToSplitDouble(fft_length,
                                  std::move(possible_comparission_data.value()));

  //Compute data and check validity
  auto possible_data =
      FullSingleFFTComputation(fft_length, weights_RE, weights_IM);
  if (!possible_data) {
    std::cout << "Error! Failed to create data." << std::endl;
    return false;
  }
  std::unique_ptr<double[]> data =
      ConvertResultsToSplitDouble(fft_length,
                                  std::move(possible_data.value()));

  double max_dev =
      GetLargestDeviation(data.get(), comparission_data.get(), fft_length);
  double avg =
      ComputeAverageDeviation(data.get(), comparission_data.get(), fft_length);
  double sigma =
      ComputeSigmaOfDeviation(data.get(), comparission_data.get(), fft_length,
                              avg);

  if (((avg > avg_deviation_threshold) ||
       (sigma > sigma_deviation_threshold)) ||
      (max_dev > max_deviation_threshold)
       ){
    std::cout << "Accuracy test failed!" << std::endl
              << "Avg: " << avg << " Threshold: "
              << avg_deviation_threshold
              << " Sigma: " << sigma << " Threshold: "
              << sigma_deviation_threshold
              << " Max Deviation: " << max_dev << " Threshold: "
              << max_deviation_threshold
              << std::endl;
    return false;
  }

  return true;
}

template <typename Integer>
bool TestFullFFT(const Integer fft_length,
                 const double avg_deviation_threshold,
                 const double sigma_deviation_threshold,
                 const double max_deviation_threshold){
  std::optional<std::string> err;

  std::string comparison_data_file_name =
    ("test_comparison_" + std::to_string(fft_length)) + ".dat";
  std::string data_file_name =
    ("test_" + std::to_string(fft_length)) + ".dat";

  err = CreateComparisonDataDouble(fft_length, comparison_data_file_name);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  err = FullSingleFFTComputation(fft_length, data_file_name);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  double avg = ComputeAverageDeviation(comparison_data_file_name,
                                       data_file_name);
  double sigma = ComputeSigmaOfDeviation(comparison_data_file_name,
                                         data_file_name, avg);
  double max_dev = GetLargestDeviation(comparison_data_file_name,
                                       data_file_name);

  if (((avg > avg_deviation_threshold) ||
       (sigma > sigma_deviation_threshold)) ||
      (max_dev > max_deviation_threshold)
       ){
    std::cout << "Accuracy test failed!" << std::endl
              << "Avg: " << avg << " Threshold: "
              << avg_deviation_threshold
              << " Sigma: " << sigma << " Threshold: "
              << sigma_deviation_threshold
              << " Max Deviation: " << max_dev << " Threshold: "
              << max_deviation_threshold
              << std::endl;
    return false;
  }

  return true;
}

template <typename Integer>
bool TestFullFFTAsynch(const Integer fft_length,
                       const int amount_of_asynch_ffts,
                       const double avg_deviation_threshold,
                       const double sigma_deviation_threshold,
                       const double max_deviation_threshold){
std::optional<std::string> err;

std::string comparison_data_file_name =
  ("test_comparison_" + std::to_string(fft_length)) + ".dat";
std::vector<std::string> data_file_name;
for(int i=0; i<amount_of_asynch_ffts; i++){
  data_file_name.push_back(((("test_" + std::to_string(fft_length)) + "_async_")
                            + std::to_string(i)) + ".dat");
}

err = CreateComparisonDataDouble(fft_length, comparison_data_file_name);
if (err) {
  std::cout << err.value() << std::endl;
  return false;
}

err = FullAsyncFFTComputation(fft_length, amount_of_asynch_ffts,
                              data_file_name);
if (err) {
  std::cout << err.value() << std::endl;
  return false;
}

double avg = ComputeAverageDeviation(data_file_name, comparison_data_file_name);
double sigma = ComputeSigmaOfDeviation(data_file_name,
                                       comparison_data_file_name, avg);
double max_dev = GetLargestDeviation(data_file_name, comparison_data_file_name);

if (((avg > avg_deviation_threshold) ||
     (sigma > sigma_deviation_threshold)) ||
    (max_dev > max_deviation_threshold)
     ){
  std::cout << "Accuracy test failed!" << std::endl
            << "Avg: " << avg << " Threshold: "
            << avg_deviation_threshold
            << " Sigma: " << sigma << " Threshold: "
            << sigma_deviation_threshold
            << " Max Deviation: " << max_dev << " Threshold: "
            << max_deviation_threshold
            << std::endl;
  return false;
}

return true;
}
