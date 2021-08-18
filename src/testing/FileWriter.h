//Contains numerous functions to write data file
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <optional>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>

#include "benchmarks/BenchUtil.h"

//Writes results of a fft that uses __half to file
template <typename Integer>
std::optional<std::string> WriteResultsToFile(const std::string file_name,
                                              const Integer fft_length,
                                              const __half* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float re = data[i];
      float im = data[i + fft_length];
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << re << " " << im << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes results of a fft that uses __half to file
template <typename Integer>
std::optional<std::string> WriteResultsREToFile(const std::string file_name,
                                                const Integer fft_length,
                                                const __half* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float re = data[i];
      myfile << re << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes results of a fft that uses 2 __half to file
template <typename Integer>
std::optional<std::string> WriteResultsToFile(const std::string file_name,
                                              const Integer fft_length,
                                              const __half* data_RE,
                                              const __half* data_IM){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float re = data_RE[i];
      float im = data_IM[i];
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << re << " " << im << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes results of a fft that uses 2 __half to file
template <typename Integer>
std::optional<std::string> WriteResultBatchToFile(
    const std::vector<std::string> file_names,
    const Integer fft_length,
    const __half* data){
  for(int j=0; j<static_cast<int>(file_names.size()); j++){
    std::ofstream myfile (file_names[j]);
    if (myfile.is_open()) {
      for(int i=0; i<fft_length; i++){
        float re = data[i + (2 * fft_length * j)];
        float im = data[i + fft_length + (2 * fft_length * j)];
        float x = static_cast<double>(i)/static_cast<double>(fft_length);
        myfile << x << " " << re << " " << im << "\n";
      }
      myfile.close();
    } else {
      return "Error! Unable to open file.";
    }
  }

  return std::nullopt;
}


//Writes results of a fft that uses __half2 to file
template <typename Integer>
std::optional<std::string> WriteResultsToFileHalf2(const std::string file_name,
                                                   const Integer fft_length,
                                                   const __half2* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " "
             << __low2float(data[i]) << " "
             << __high2float(data[i])
             << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes results of a fft that uses cufftDoubleComplex to file
template <typename Integer>
std::optional<std::string> WriteResultsToFileDouble2(
    const std::string file_name,
    const Integer fft_length,
    const cufftDoubleComplex* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << data[i].x
             << " " << data[i].y
             << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes Benchmark results to file
template <typename Integer>
std::optional<std::string> WriteBenchResultsToFile(
    const std::vector<double> run_avg,
    const std::vector<double> run_sig,
    const std::vector<Integer> fft_length,
    const std::string sample_size){
  std::ofstream myfile ("BenchResults_samples_" + sample_size + ".dat");
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(fft_length.size()); i++){
      myfile << fft_length[i] << " "
             << run_avg[i] << " "
             << run_sig[i] << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes error message to file
std::optional<std::string> WriteLogToFile(const std::string file_name,
                                          const std::string error_mess){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    myfile << error_mess << "\n";
    myfile.close();
  } else {
    std::cout << "Unable to open file " << file_name << std::endl;
  }
  return std::nullopt;
}

//Writes result of Accuracy test to file
template <typename Integer>
std::optional<std::string> WriteAccuracyTestResultsToFile(
    const std::vector<double> average,
    const std::vector<double> std_dev,
    const int sample_size,
    const std::vector<Integer> fft_length){
  std::ofstream myfile ("Accuracy_Test.dat");
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(average.size()); i++){
      myfile << sample_size << " "
             << fft_length[i] << " "
             << average[i] << " "
             << std_dev[i] << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

template <typename Integer>
std::optional<std::string> WriteTunerDataToFile(
      const std::vector<RunResults> results,
      const Integer fft_length){
  std::ofstream myfile ((("TunerData_" + std::to_string(fft_length)) + ".dat"));
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(results.size()); i++){
      int mode = results[i].mode_ == Mode_256 ? 256 : 4096;
      myfile << fft_length << " "
             << mode << " "
             << results[i].base_fft_warps_per_block_ << " "
             << results[i].r16_warps_per_block_ << " "
             << results[i].r2_blocksize_ << " "
             << results[i].results_.average_time_ << " "
             << results[i].results_.std_deviation_ << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

template <typename Integer>
std::optional<std::string> WriteTunerDataToFile(
      const std::vector<RunResults> results,
      const std::vector<Integer> fft_length){
  std::ofstream myfile ("TunerResults.dat");
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(results.size()); i++){
      int mode = results[i].mode_ == Mode_256 ? 256 : 4096;
      myfile << fft_length[i] << " "
             << mode << " "
             << results[i].base_fft_warps_per_block_ << " "
             << results[i].r16_warps_per_block_ << " "
             << results[i].r2_blocksize_ << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

template <typename Integer>
std::optional<std::string> WriteBenchResultsToFile(
      const std::vector<RunResults> results,
      const std::vector<Integer> fft_length){
  std::ofstream myfile ("BenchResults.dat");
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(results.size()); i++){
      int mode = results[i].mode_ == Mode_256 ? 256 : 4096;
      myfile << fft_length[i] << " "
             << mode << " "
             << results[i].base_fft_warps_per_block_ << " "
             << results[i].r16_warps_per_block_ << " "
             << results[i].r2_blocksize_ << " "
             << results[i].results_.average_time_ << " "
             << results[i].results_.std_deviation_ << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

template <typename Integer>
std::optional<std::string> WriteBenchResultsToFile(
      const std::vector<BenchResult> results,
      const std::vector<Integer> fft_length){
  std::ofstream myfile ("BenchResults.dat");
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(results.size()); i++){
      myfile << fft_length[i] << " "
             << results[i].average_time_ << " "
             << results[i].std_deviation_ << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}
