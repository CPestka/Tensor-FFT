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
#include "../unitTesting/FFTTest.cu"
#include "../../base/ComputeFFT.h"
#include "../FileWriter.h"

int main(){
  constexpr int fft_length = 16*16*16*16*16;

  constexpr int max_frequency_cutof = fft_length;
  constexpr int min_frequency_cutof = 1;
  constexpr int seed = 42;

  std::vector<float> weights_RE =
      GetRandomWeights(max_frequency_cutof, seed);
  std::vector<float> weights_IM =
      GetRandomWeights(max_frequency_cutof, seed * seed);

  std::vector<int> frequency_cutof;
  std::vector<double> avg_dev;
  std::vector<double> sigma_of_dev;
  std::vector<double> max_dev;

  std::optional<std::string> err;

  frequency_cutof.push_back(min_frequency_cutof);
  while (frequency_cutof.back() <= max_frequency_cutof &&
         frequency_cutof.back() != 0) {
    std::cout << "Testing frequency cutof: " << frequency_cutof.back()
              << std::endl;

    //Compute comparision data and check validity
    auto possible_comparission_data =
        CreateComparisonDataDouble(fft_length, weights_RE, weights_IM,
                                   frequency_cutof.back());
    if (!possible_comparission_data) {
      std::cout << "Error! Failed to create comparision data." << std::endl;
      return false;
    }
    std::unique_ptr<double[]> comparission_data = ConvertResultsToSplitDouble(
        fft_length, std::move(possible_comparission_data.value()));

    //Compute data and check validity
    auto possible_data =
        FullSingleFFTComputation(fft_length, weights_RE, weights_IM,
                                   frequency_cutof.back());
    if (!possible_data) {
      std::cout << "Error! Failed to create data." << std::endl;
      return false;
    }
    std::unique_ptr<double[]> data =
        ConvertResultsToSplitDouble(fft_length,
                                    std::move(possible_data.value()));

    max_dev.push_back(GetLargestDeviation(data.get(),
                                          comparission_data.get(),
                                          fft_length));
    avg_dev.push_back(ComputeAverageDeviation(data.get(),
                                              comparission_data.get(),
                                              fft_length));
    sigma_of_dev.push_back(ComputeSigmaOfDeviation(data.get(),
                                                   comparission_data.get(),
                                                   fft_length,
                                                   avg_dev.back()));

    frequency_cutof.push_back(frequency_cutof.back() * 2);
  }

  err = WriteAccuracyTestResultsToFile(avg_dev, sigma_of_dev, max_dev,
                                       frequency_cutof);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  return true;
}
