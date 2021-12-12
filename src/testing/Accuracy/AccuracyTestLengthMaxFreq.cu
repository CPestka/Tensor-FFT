//Accuracy benchmark for a varrying length but with fixed frequency bandwidth.
#include <vector>
#include <memory>

#include "../DataMaker.cu"
#include "../WeightMaker.h"
#include "ComputeError.h"
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
  cudaMalloc(&dptr_data, sizeof(cufftDoubleComplex) * fft_length);
  //Produce input data based on weights
  SineSupperposition<int,cufftDoubleComplex><<<fft_length / 1024, 1024>>>(
      fft_length, dptr_data, dptr_weights, amount_of_frequencies, 1.0);

  std::unique_ptr<cufftDoubleComplex[]> data =
      std::make_unique<cufftDoubleComplex[]>(fft_length);

  cudaMemcpy(data.get(), dptr_data, fft_length * sizeof(cufftDoubleComplex),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(dptr_data);

  return (MaxValue<Integer,cufftDoubleComplex>(data.get(), fft_length) /
          normalization_target);
}

int main(){
  int fft_length_min_log2 = 12;
  int fft_length_max_log2 = 28;
  int max_amount_of_frequencies = ExactPowerOf2<int>(fft_length_max_log2);
  std::vector<double> normalize_to;

  std::unique_ptr<float2[]> weights =
      std::make_unique<float2[]>(max_amount_of_frequencies);
  SetRandomWeights(weights.get(), max_amount_of_frequencies, 42*42);
  float2* dptr_weights = nullptr;
  cudaMalloc(&dptr_weights, sizeof(float2) * max_amount_of_frequencies);
  cudaMemcpy(dptr_weights, weights.get(),
             sizeof(float2) * max_amount_of_frequencies, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  std::vector<int> fft_lengths;
  std::vector<Errors> errors;
  std::vector<int> amount_of_frequencies_vec;
  std::vector<double> normalization_factors;

  for(int i=fft_length_min_log2; i<=fft_length_max_log2; i++){
    normalize_to.push_back(1.0);
    fft_lengths.push_back(ExactPowerOf2<int>(i));
    amount_of_frequencies_vec.push_back(fft_lengths.back());
    std::cout << fft_lengths.back() << std::endl;
    double normalization_factor =
        GetNormalizationFactor<int>(normalize_to.back(), dptr_weights,
                                    amount_of_frequencies_vec.back(), fft_lengths.back());
    normalization_factors.push_back(normalization_factor);
    errors.push_back(ComputeOurVsFp64Errors<int>(static_cast<int>(fft_lengths.back()),
        dptr_weights, amount_of_frequencies_vec.back(), normalization_factor));

  }

  WriteAccuracyToFile("AccTest_our_N_nu_max.dat", normalize_to, fft_lengths, errors,
                      amount_of_frequencies_vec, normalization_factors);

  cudaFree(dptr_weights);
}
