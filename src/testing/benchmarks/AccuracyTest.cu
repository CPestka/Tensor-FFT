//Used to measure
#include <iostream>
#include <string>
#include <optional>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../AccuracyCalculator.h"
#include "../unitTesting/CuFFTTest.h"
#include "../unitTesting/FTTTest.cu"
#include "../../base/ComputeFFT.h"
#include "../FileWriter.h"

int main(){
  constexpr int sample_size = 10;
  constexpr int log_fft_length_boundry = 29;

  std::vector<int> fft_length;
  std::vector<double> avg_dev;
  std::vector<double> sigma_of_dev;

  std::optional<std::string> err;

  int tmp_length = 16*8;
  for(int i=8; i<=log_fft_length_boundry; i++){
    tmp_length = tmp_length * 2;
    fft_length.push_back(tmp_length);

    std::string cuFFT_file_name = ("accuracy_cuFFT_" +
                                   std::to_string(fft_length.back()))
                                  + ".dat";
    err = CreateComparisonDataDouble(fft_length.back(), cuFFT_file_name);
    if (err) {
      std::cout << err.value() << std::endl;
      return false;
    }

    std::vector<std::string> fft_results_file_names;
    for(int k=0; k<sample_size; k++){
      fft_results_file_names.push_back(((("accuracy_" +
                                        std::to_string(fft_length.back())) +
                                        "_") + std::to_string(k)) + ".dat");
      err = FullSingleFFTComputation(fft_length.back(),
                                     fft_results_file_names.back());
      if (err) {
        std::cout << err.value() << std::endl;
        return false;
      }
    }

    avg_dev.push_back(ComputeAverageDeviation(fft_results_file_names,
                                              cuFFT_file_name));
    sigma_of_dev.push_back(ComputeSigmaOfDeviation(fft_results_file_names,
                                                   cuFFT_file_name,
                                                   avg_dev.back()));
  }

  err = WriteAccuracyTestResultsToFile(avg_dev, sigma_of_dev, sample_size,
                                       fft_length);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  return true;
}
