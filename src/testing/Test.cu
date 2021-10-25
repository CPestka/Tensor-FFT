
#include <iostream>
#include <memory>

#include "../base/Plan.h"
#include "../base/ComputeFFT.h"
#include "WeightMaker.h"
#include "DataMaker.cu"
#include "Accuracy/ComputeError.h"
#include "FileWriter.h"

int main(){
  int fft_length = 16*16*16;
  //int amount_of_frequencies = 256;
  int amount_of_frequencies = 5;

  std::unique_ptr<float2[]> weights =
      std::make_unique<float2[]>(amount_of_frequencies);
  //SetRandomWeights(weights.get(), amount_of_frequencies, 42*42);
  SetDummyWeights(weights.get());
  float2* dptr_weights = nullptr;
  cudaMalloc(&dptr_weights, sizeof(float2) * amount_of_frequencies);
  cudaMemcpy(dptr_weights, weights.get(),
             sizeof(float2) * amount_of_frequencies, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  Errors err = ComputeOurVsFp64Errors<int>(fft_length, dptr_weights,
                                           amount_of_frequencies, 1.0);

  std::cout << "Max div: " << err.MaxDiv
            << " MAE: " << err.MeanAbsoluteError
            << " RMSE: " << err.RootMeanSquareError << std::endl;

  std::unique_ptr<__half2[]> results =
      GetOurFP16Data<int>(dptr_weights, amount_of_frequencies, fft_length, 1.0);

  WriteFFTToFile("TestResults.dat", fft_length, results.get());

  cudaFree(dptr_weights);
}
