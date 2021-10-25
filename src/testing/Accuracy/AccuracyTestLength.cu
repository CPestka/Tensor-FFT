//Accuracy benchmark for a varrying length but with fixed frequency bandwidth.

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

  std::unique_ptr<cufftDoubleComplex> data =
      std::make_unique<cufftDoubleComplex>(fft_length);

  cudaMemcpy(data.get(), dptr_data, fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  return (normalization_target /
          MaxValue<Integer,cufftDoubleComplex>(data.get(), fft_length));
}

int main(){
  int fft_length_min_log2 = 12;
  int fft_length_max_log2 = 28;
  int amount_of_frequencies = 256;
  double normalize_to = 1.0;

  std::unique_ptr<float2> weights =
      std::make_unique<float2>(amount_of_frequencies);
  SetRandomWeights(weights.get(), amount_of_frequencies, 42*42);
  float2* dptr_weights = nullptr;
  cudaMalloc(&dptr_weights, sizeof(float2) * amount_of_frequencies);
  cudaMemcpy(dptr_weights, weights.get(),
             sizeof(float2) * amount_of_frequencies, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  double normalization_factor =
      GetNormalizationFactor<int>(normalize_to, weights.get(),
                                  amount_of_frequencies);

  std::vector<int64_t> fft_lengths;
  std::vector<Errors> errors;
  std::vector<int> amount_of_frequencies_vec;

  for(int i=fft_length_max_log2; i<=fft_length_min_log2; i++){
    fft_lengths.push_back(ExactPowerOf2(i));
    errors.push_back(ComputeOurVsFp64Errors(fft_lengths.back(), dptr_weights,
        amount_of_frequencies, normalization_factor));
    amount_of_frequencies_vec.push_back(amount_of_frequencies);
  }

  WriteAccuracyToFile("AccuracyTest.dat", fft_lengths, normalize_to, errors,
                      amount_of_frequencies_vec);

  cudaFree(dptr_weights);
}