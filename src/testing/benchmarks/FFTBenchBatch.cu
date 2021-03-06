//Used to benchmark the function ComputeFFT
#include <iostream>
#include <vector>
#include <optional>
#include <string>

#include "../FileWriter.h"
#include "Bench.h"

int main(){
  constexpr int start_fft_length = 16*16;
  constexpr int end_fft_length = 16*16*16*16*16 * 16*2*2*2;

  constexpr int sample_size = 200;
  constexpr int warmup_samples = 5;
  constexpr int async_batch_size = 20;

  std::vector<int> fft_length;
  fft_length.push_back(start_fft_length);

  std::vector<RunResults> bench_data;

  std::optional<std::string> err;

  while (fft_length.back() <= end_fft_length) {
    std::optional<RunResults> tmp =
      Benchmark(fft_length.back(), warmup_samples, sample_size,
                async_batch_size, "TunerResultsAsync.dat");
      //Benchmark(fft_length.back(), warmup_samples,
      //          sample_size, async_batch_size, 256, 8, 8, 256);
    if (tmp) {
      bench_data.push_back(tmp.value());
      fft_length.push_back(fft_length.back() * 2);
    }
  }

  WriteBenchResultsToFile(bench_data, fft_length);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  return true;
}
