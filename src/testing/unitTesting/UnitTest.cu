//Used to test functonal correctness of out ffts
#include <iostream>
#include <string>

#include "FFTTest.cu"

int main(){
  constexpr int start_fft_length = 16*16;
  constexpr int end_fft_length = 16*16*16*16*16;
  constexpr int async_batch_size = 4;
  constexpr double average_deviation_threshold = 0.0001;
  constexpr double sigma_deviation_threshold = 0.001;
  constexpr double max_deviation_threshold = 0.005;

  int fft_length = start_fft_length;
  while (fft_length <= end_fft_length) {
    if ((!TestFullFFT(fft_length, average_deviation_threshold,
                      sigma_deviation_threshold, max_deviation_threshold))
         || (!TestFullFFTAsynch(fft_length, async_batch_size,
                                average_deviation_threshold,
                                sigma_deviation_threshold,
                                max_deviation_threshold))
        ) {
      std::cout << "Error! Test at fft_length: "
                << fft_length
                << " failed!"
                << std::endl;
      return false;
    }

    fft_length = (fft_length * 2);
  }

  std::cout << "All tests passed!" <<std::endl;

  return true;
}
