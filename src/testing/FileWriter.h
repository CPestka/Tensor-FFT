//Contains numerous functions to write data file
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <optional>

#include "Accuracy/ComputeError.h"

struct BatchResult{
  double Average_;
  double RMS_;
  int fft_length_;
};

template <typename float2_t>
std::optional<std::string> WriteFFTToFile(const std::string file_name,
                                          const int64_t fft_length,
                                          const float2_t* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int64_t i=0; i<fft_length; i++){
      double re = data[i].x;
      double im = data[i].y;
      double x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << re << " " << im << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

std::optional<std::string> WriteAccuracyToFile(
    const std::string file_name,
    const std::vector<double> normalized_to,
    const std::vector<int> fft_length,
    const std::vector<Errors> errors,
    const std::vector<int> amount_of_frequencies){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(fft_length.size()); i++){
      myfile << fft_length[i] << " "
             << normalized_to[i] << " "
             << amount_of_frequencies[i] << " "
             << errors[i].MaxDiv << " "
             << errors[i].MeanAbsoluteError << " "
             << errors[i].RootMeanSquareError << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}

std::optional<std::string> WriteBenchResultsToFile(
    const std::string file_name,
    const std::vector<BatchResult> data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(data.size()); i++){
      myfile << data[i].fft_length_ << " "
             << data[i].Average_ << " "
             << data[i].RMS_ << "\n";
    }
    myfile.close();
  } else {
    return "Error! Unable to open file.";
  }
  return std::nullopt;
}
