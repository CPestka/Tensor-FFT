//Accuracy benchmark for a varrying frequency bandwidth but with fixed length.
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

  return (MaxValue<Integer,cufftDoubleComplex>(data.get(), fft_length) /
          normalization_target);
}

int main(){
  int fft_length = 16*16*16*16*16;
  int max_frequencies_log2 = 20;

  double normalize_to = 1.0;

  std::vector<int> fft_lengths;
  std::vector<Errors> errors;
  std::vector<int> amount_of_frequencies_vec;

  std::unique_ptr<float2[]> weights =
      std::make_unique<float2[]>(ExactPowerOf2<int>(max_frequencies_log2));
  float2* dptr_weights = nullptr;
  cudaMalloc(&dptr_weights, sizeof(float2) *
                            ExactPowerOf2<int>(max_frequencies_log2));

  for(int i=1; i<=max_frequencies_log2; i++){
    fft_lengths.push_back(fft_length);
    amount_of_frequencies_vec.push_back(ExactPowerOf2<int>(i));
    std::cout << amount_of_frequencies_vec.back() << std::endl;

    SetRandomWeights(weights.get(), amount_of_frequencies_vec.back(), 42*42);
    cudaMemcpy(dptr_weights, weights.get(),
               sizeof(float2) * amount_of_frequencies_vec.back(),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    double normalization_factor =
        GetNormalizationFactor<int>(normalize_to, dptr_weights,
                                    amount_of_frequencies_vec.back(),
                                    fft_length);

    std::cout << normalization_factor << std::endl;

    errors.push_back(ComputeOurVsFp64Errors<int>(fft_lengths.back(),
        dptr_weights, amount_of_frequencies_vec.back(), normalization_factor));
  }

  WriteAccuracyToFile("AccTest_our_nu.dat", normalize_to, fft_lengths, errors,
                      amount_of_frequencies_vec);

  cudaFree(dptr_weights);
}
