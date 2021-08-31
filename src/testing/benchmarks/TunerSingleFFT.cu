//Used to benchmark the function ComputeFFT
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include "Bench.h"
#include "../FileWriter.h"

int main(){
  constexpr int start_fft_length = 16*16;
  constexpr int end_fft_length = 16*16*16*16*16 * 16*8;

  constexpr int sample_size = 100;
  constexpr int warmup_samples = 5;

  constexpr int device_id = 0; //To tune to device

  std::vector<RunConfig> optimal_config;

  std::vector<int> fft_length;
  fft_length.push_back(start_fft_length);

  std::optional<std::string> err;

  while (fft_length.back() <= end_fft_length) {
    std::cout << "Current fft_length: " << fft_length.back() << std::endl;

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

  err = WriteTunerResultsToFile(optimal_config, fft_length);
  if (err) {
    std::cout << err.value() << std::endl;
    return false;
  }

  return true;
}
