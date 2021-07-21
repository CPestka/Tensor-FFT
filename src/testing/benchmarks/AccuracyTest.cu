#pragma once
#include <iostream>
#include <string>
#include <optional>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../AccuracyCalculator.cu"
#include "../unitTesting/CuFFTTest.cu"
#include "../../base/ComputeFFT.cu"
#include "../FileWriter.cu"

int main(){
  int sample_size = 4;
  int fft_length_boundry = 5;

  std::vector<int> fft_length;
  std::vector<double> avg_dev;
  std::vector<double> sigma_of_dev;

  std::optional<std::string> err;

  int length_16 = 16;
  for(int i=2; i<6; i++){
    length_16 = length_16 * 16;
    int length_2;
    for(int j=0; j<3; j++){
      if (j == 0) {
        length_2 = 1;
      } else {
        length_2 = length_2 * 2;
      }
      fft_length.push_back(length_16 * length_2);

      std::string cuFFT_file_name = ("accuracy_cuFFT_" + fft_length.back())
                                    + ".dat";
      err = CreateComparisionData(fft_length.back(), cuFFT_file_name);
      if (err) {
        return err;
      }

      std::vector<std::string> fft_results_file_names;
      for(int k=0; k<sample_size; k++){
        fft_results_file_names.push_back(("accuracy_" + fft_length.back())
                                          + ".dat");
        err = FullSingleFFTComputation(fft_length.back(),
                                       fft_results_file_names.back());
        if (err) {
          return err;
        }
      }

      avg_dev.push_back(ComputeAverageDeviation(fft_results_file_names,
                                                cuFFT_file_name));
      sigma_of_dev.push_back(ComputeSigmaOfDeviation(fft_results_file_names,
                                                     cuFFT_file_name,
                                                     avg_dev.back()));
    }
  }

  err = WriteAccuracyTestResultsToFile(avg_dev, sigma_of_dev, sample_size,
                                       fft_length);
  if (err) {
    return err;
  }

  return true;
}
