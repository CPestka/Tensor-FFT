//Used to benchmark the function ComputeFFT
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.cu"
#include "FileWriter.cu"
#include "Timer.h"
#include "../base/ComputeFFT.cu"
#include "../base/Plan.cpp"

double ComputeAverage(std::vector<double> data){
  double tmp = 0;
  for(int i=0; i<static_cast<int>(data.size()); i++){
    tmp += data[i];
  }
  return (tmp / (static_cast<double>(data.size()) - 1));
}

double GetAverageExecutionTime(int fft_length, int warmup_samples,
                               int sample_size, int r16_warp_amount,
                               int dft_warp_amount, int r2_blocksize,
                               int transpose_blocksize){
  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostion(fft_length,  weights);

  std::vector<double> runtime;

  Plan my_plan;
  if (CreatePlan(fft_length, transpose_blocksize, dft_warp_amount,
                 r16_warp_amount, r2_blocksize)) {
    my_plan = CreatePlan(fft_length).value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  std::optional<std::string> error_mess;

  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  for(int k=0; k<sample_size + warmup_samples; k++){
    error_mess = my_handler.CopyDataHostToDevice(data.get());
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    IntervallTimer computation_time;
    error_mess = ComputeFFT(my_plan, my_handler);
    if (error_mess) {
      std::cout << error_mess.value() << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    if (k >= warmup_samples) {
      runtime.push_back(computation_time.getTimeInNanoseconds());
    }
  }

  return ComputeAverage(runtime);
}

int main(){
  int log_length_max = 29;
  int sample_size = 20;
  int warmup_samples = 4;
  int warp_max_cap = 32;

  long long fft_length = 16 * 8;

  std::vector<int> fastest_transpose_blocksizes;
  std::vector<int> fastest_dft_warp_counts;
  std::vector<int> fastest_r16_warp_counts;
  std::vector<int> fastest_R2_blocksizes;

  for(int i=8; i<=log_length_max; i++){
    fft_length = fft_length * 2;
    std::cout << "Starting fft length: " << fft_length << std::endl;

    int warp_amount_cap = (warp_max_cap * 256) > fft_length ?
                          (fft_length / 256) : warp_max_cap;

    double runtime = 0;
    double tmp = 0;
    int fastest_warp_count = 0;

    //Determine fastest r16 warp count
    for(int j=1; j<=warp_amount_cap; j=j*2){
      if (j == 1) {
        runtime = GetAverageExecutionTime(
          fft_length, warmup_samples,sample_size , j,
          warp_amount_cap < 4 ? warp_amount_cap : 4, 256,
          fft_length < 1024 ? fft_length : 1024);
        fastest_warp_count = j;
      } else {
        tmp = GetAverageExecutionTime(
          fft_length, warmup_samples, sample_size, j,
          warp_amount_cap < 4 ? warp_amount_cap : 4, 256,
          fft_length < 1024 ? fft_length : 1024);
        if (tmp < runtime) {
          runtime = tmp;
          fastest_warp_count = j;
        }
      }
    }

    fastest_r16_warp_counts.push_back(fastest_warp_count);

    //Determine fastest dft warp count
    for(int j=1; j<=warp_amount_cap; j=j*2){
      if (j == 1) {
        runtime = GetAverageExecutionTime(
            fft_length, warmup_samples, sample_size,
            fastest_r16_warp_counts.back(), j, 256,
            fft_length < 1024 ? fft_length : 1024);
        fastest_warp_count = j;
      } else {
        tmp = GetAverageExecutionTime(
            fft_length, warmup_samples, sample_size,
            fastest_r16_warp_counts.back(), j, 256,
            fft_length < 1024 ? fft_length : 1024);
        if (tmp < runtime) {
          runtime = tmp;
          fastest_warp_count = j;
        }
      }
    }

    fastest_dft_warp_counts.push_back(fastest_warp_count);

    //Determine fastest R2 blocksize
    for(int j=1; j<=warp_amount_cap; j=j*2){
      if (j == 1) {
        runtime = GetAverageExecutionTime(
            fft_length, warmup_samples, sample_size,
            fastest_r16_warp_counts.back(), fastest_dft_warp_counts.back(),
            j * 32, fft_length < 1024 ? fft_length : 1024);
        fastest_warp_count = j;
      } else {
        tmp = GetAverageExecutionTime(
            fft_length, warmup_samples, sample_size,
            fastest_r16_warp_counts.back(), fastest_dft_warp_counts.back(),
            j * 32, fft_length < 1024 ? fft_length : 1024);
        if (tmp < runtime) {
          runtime = tmp;
          fastest_warp_count = j;
        }
      }
    }

    fastest_R2_blocksizes.push_back(fastest_warp_count * 32);

    //Determine fastest transpose blocksize
    for(int j=1; j<=warp_amount_cap; j=j*2){
      if (j == 1) {
        runtime = GetAverageExecutionTime(
            fft_length, warmup_samples, sample_size,
            fastest_r16_warp_counts.back(), fastest_dft_warp_counts.back(),
            fastest_R2_blocksizes.back(), j * 32);
        fastest_warp_count = j;
      } else {
        tmp = GetAverageExecutionTime(
            fft_length, warmup_samples, sample_size,
            fastest_r16_warp_counts.back(), fastest_dft_warp_counts.back(),
            fastest_R2_blocksizes.back(), j * 32);
        if (tmp < runtime) {
          runtime = tmp;
          fastest_warp_count = j;
        }
      }
    }

    fastest_transpose_blocksizes.push_back(fastest_warp_count * 32);
  }

  std::cout << WriteTunerResultsToFile(
      fastest_transpose_blocksizes, fastest_dft_warp_counts,
      fastest_r16_warp_counts,
      fastest_R2_blocksizes).value_or("Tuner finished successfully")
            << std::endl;

  return true;
}
