//Contains numerous functions to write data file
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <optional>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

//Writes results of a fft that uses __half to file
std::optional<std::string> WriteResultsToFile(std::string file_name,
                                              int fft_length, __half* data){
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
std::optional<std::string> WriteResultsREToFile(std::string file_name,
                                                int fft_length, __half* data){
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
std::optional<std::string> WriteResultsToFile(std::string file_name,
                                              int fft_length, __half* data_RE,
                                              __half* data_IM){
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


//Writes results of a fft that uses __half2 to file
std::optional<std::string> WriteResultsToFileHalf2(std::string file_name,
                                                   int fft_length,
                                                   __half2* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << __low2float(data[i]) << " " << __high2float(data[i])
             << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

//Writes Benchmark results to file
std::optional<std::string> WriteBenchResultsToFile(std::vector<double> run_avg,
                                                   std::vector<double> run_sig,
                                                   std::vector<int> fft_length,
                                                   std::string sample_size){
  std::ofstream myfile ("BenchResults" + sample_size + ".dat");
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
std::optional<std::string> WriteLogToFile(std::string file_name,
                                          std::string error_mess){
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
std::optional<std::string> WriteAccuracyTestResultsToFile(
    std::vector<double> average, std::vector<double> std_dev, int sample_size,
    std::vector<int> fft_length){
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
