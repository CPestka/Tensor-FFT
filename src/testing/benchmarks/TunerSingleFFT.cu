//Used to benchmark the function ComputeFFT
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include "Bench.h"
#include "FileWriter.cu"

int main(){
  constexpr int start_fft_length = 16*16;
  constexpr int end_fft_length = 16*16*16*16*16*16*2*2*2;

  constexpr int sample_size = 200;
  constexpr int warmup_samples = 5;

  constexpr int device_id = 0; //To tune to device

  std::vector<RunConfig> optimal_config;

  std::vecotor<int> fft_length
  fft_length.push_back(start_fft_length);

  std::optional<std::string> err;

  while (fft_length.back() <= end_fft_length) {
    RunParameterSearchSpace search_space =
        GetSearchSpace(fft_length.back(), device_id);

    std::vector<RunConfig> configs = GetRunConfigs(search_space);

    std::vector<RunResults> bench_data =
        RunBenchOverSearchSpace(configs, sample_size, warmup_samples,
                                fft_length.back());

    err = WriteTunerDataToFile(bench_data, fft_length.back());
    if (err) {
      std::cout << err.value() << std::endl;
      return false;
    }

    optimal_config.push_back(GetFastestConfig(bench_data));

    fft_length.push_back(fft_length.back() * 2);
  }

  err = WriteTunerResultsToFile(fft_length, optimal_config);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  return true;
}
