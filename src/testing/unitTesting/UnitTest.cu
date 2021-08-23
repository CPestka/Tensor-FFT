//Used to test functonal correctness of out ffts
#include <iostream>
#include <string>

#include "FFTTest.cu"

int main(){
  constexpr int start_fft_length = 16*16;
  constexpr int end_fft_length = 16*16*16*16*16;

  constexpr int runs_per_fft_length = 10;
  constexpr int highest_harmonic = 50;

  constexpr double average_deviation_threshold = 0.0001;
  constexpr double sigma_deviation_threshold = 0.001;
  constexpr double max_deviation_threshold = 0.005;

  std::vector<std::vector<float>> weights_RE;
  std::vector<std::vector<float>> weights_IM;

  for(int i=0; i<runs_per_fft_length; i++){
    weights_RE.push_back(GetRandomWeights(highest_harmonic, 42 * i));
    weights_IM.push_back(GetRandomWeights(highest_harmonic, 42 * 42 * i));
  }

  int fft_length = start_fft_length;
  int j = 0;
  while (fft_length <= end_fft_length) {
    if (!TestFullFFT(fft_length,
                     average_deviation_threshold,
                     sigma_deviation_threshold,
                     max_deviation_threshold,
                     weights_RE[j],
                     weights_IM[j])) {
      std::cout << "Error! Test at fft_length: "
                << fft_length
                << " failed!"
                << std::endl;
      return false;
    }

    fft_length = (fft_length * 2);
    j++;
  }

  std::cout << "All tests passed!" <<std::endl;

  return true;
}
