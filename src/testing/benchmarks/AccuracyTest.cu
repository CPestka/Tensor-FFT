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
  constexpr int start_fft_length = 16*16;
  constexpr int end_fft_length = 16*16*16*16*16;

  constexpr int highest_harmonic = end_fft_length;
  constexpr int seed = 42;

  std::vector<float> weights_RE =
      GetRandomWeights(highest_harmonic, seed);
  std::vector<float> weights_IM =
      GetRandomWeights(highest_harmonic, seed * seed);

  std::vector<int> fft_length;
  std::vector<double> avg_dev;
  std::vector<double> sigma_of_dev;
  std::vector<double> max_dev;

  std::optional<std::string> err;

  fft_length.push_back(start_fft_length);
  while (fft_length.back() <= end_fft_length && fft_length.back() != 0) {
    std::cout << "Testing fft length: " << fft_length.back() << std::endl;

    //Compute comparision data and check validity
    auto possible_comparission_data =
        CreateComparisonDataDouble(fft_length.back(), weights_RE, weights_IM);
    if (!possible_comparission_data) {
      std::cout << "Error! Failed to create comparision data." << std::endl;
      return false;
    }
    std::unique_ptr<double[]> comparission_data = ConvertResultsToSplitDouble(
        fft_length.back(), std::move(possible_comparission_data.value()));

    //Compute data and check validity
    auto possible_data =
        FullSingleFFTComputation(fft_length.back(), weights_RE, weights_IM);
    if (!possible_data) {
      std::cout << "Error! Failed to create data." << std::endl;
      return false;
    }
    std::unique_ptr<double[]> data =
        ConvertResultsToSplitDouble(fft_length.back(),
                                    std::move(possible_data.value()));

    max_dev.push_back(GetLargestDeviation(data.get(),
                                          comparission_data.get(),
                                          fft_length.back()));
    avg_dev.push_back(ComputeAverageDeviation(data.get(),
                                              comparission_data.get(),
                                              fft_length.back()));
    sigma_of_dev.push_back(ComputeSigmaOfDeviation(data.get(),
                                                   comparission_data.get(),
                                                   fft_length.back(),
                                                   avg_dev.back()));

    fft_length.push_back(fft_length.back() * 2);
  }

  err = WriteAccuracyTestResultsToFile(avg_dev, sigma_of_dev, max_dev,
                                       fft_length);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  return true;
}
