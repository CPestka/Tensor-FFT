//Accuracy benchmark for a varrying frequency bandwidth but with fixed length.
#pragma once

#include <vector>
#include <memory>

#include "../DataMaker.cu"
#include "../ComputeError.h"
#include "../FileWriter.h"

template <typename Integer>
Integer ExactPowerOf2(const int exponent){
  if (exponent < 0) {
    std::cout << "Error! Negative exponent not allowed." << std::endl;
  }

  Integer result = 1;
  for(int i=0; i<exponent; i++){
    result *=2;
  }
  return result;
}

template <typename Integer>
double GetNormalizationFactor(double normalization_target, float2* dptr_weights,
                              int amount_of_frequencies, Integer fft_length){
  cufftDoubleComplex* dptr_data;
  cudaMalloc(&dptr_input_data, sizeof(cufftDoubleComplex) * fft_length);
  //Produce input data based on weights
  SineSupperposition<cufftDoubleComplex><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_data, dptr_weights, amount_of_frequencies, 1.0);

  std::unique_ptr<cufftDoubleComplex[]> data =
      std::make_unique<cufftDoubleComplex[]>(fft_length);

  cudaMemcpy(data.get(), dptr_data, fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  return (normalization_target /
          MaxValue<Integer,cufftDoubleComplex>(data.get(), fft_length));
}

int main(){
  int fft_length = 16*16*16*16*16;
  int max_frequencies = fft_length;
  int frequency_steps = 256;
  int frequency_increment = fft_length / frequency_steps;

  double normalize_to = 1.0;

  std::vector<int64_t> fft_lengths;
  std::vector<Errors> errors;
  std::vector<int> amount_of_frequencies_vec;

  std::unique_ptr<float2[]> weights =
      std::make_unique<float2[]>(max_frequencies);
  float2* dptr_weights = nullptr;
  cudaMalloc(&dptr_weights, sizeof(float2) * max_frequencies);


  for(int i=1; i<=frequency_steps; i++){
    SetRandomWeights(weights.get(), frequency_increment * i, 42*42);
    cudaMemcpy(dptr_weights, weights.get(),
               sizeof(float2) * frequency_increment * i,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    double normalization_factor =
        GetNormalizationFactor<int>(normalize_to, weights.get(),
                                    frequency_increment * i);

    fft_lengths.push_back(fft_length);
    amount_of_frequencies_vec.push_back(frequency_increment * i);

    errors.push_back(ComputeOurVsFp64Errors(fft_lengths.back(), dptr_weights,
        frequency_increment * i, normalization_factor));

  }

  WriteAccuracyToFile("AccuracyTest.dat", fft_lengths, normalize_to, errors,
                      amount_of_frequencies_vec);

  cudaFree(dptr_weights);
}
