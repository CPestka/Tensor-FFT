//
#include <iostream>
#include <memory>
#include <vector>
#include <assert.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>

#include "../base/Timer.h"
#include "FileWriter.cu"
#include "TestingDataCreation.cu"


int main(){
  long long fft_length = 16*16*16;
  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half2[]> data =
      CreateSineSuperpostionH2(fft_length,  weights);
  weights.clear();
  weights.push_back(0.0);
  std::unique_ptr<__half2[]> results =
      CreateSineSuperpostionH2(fft_length,  weights);

  __half2* dptr_data;
  __half2* dptr_results;
  cudaMalloc(&dptr_data, sizeof(__half2) * fft_length);
  cudaMalloc(&dptr_results, sizeof(__half2) * fft_length);
  cudaMemcpy(dptr_data, data.get(), fft_length * sizeof(__half2),
             cudaMemcpyHostToDevice);

  WriteResultsToFileHalf2("cuFFTinput.dat", fft_length, data.get());

  cufftHandle plan;
  cufftResult r;
  r = cufftCreate(&plan);
  assert(r == CUFFT_SUCCESS);
  size_t size = 0;
  r = cufftXtMakePlanMany(plan, 1, &fft_length, nullptr, 1, 1, CUDA_C_16F,
                          nullptr, 1, 1, CUDA_C_16F, 1, &size, CUDA_C_16F);
  assert(r == CUFFT_SUCCESS);
  IntervallTimer timer;
  r = cufftXtExec(plan, dptr_data, dptr_results, CUFFT_FORWARD);
  assert(r == CUFFT_SUCCESS);
  std::cout << "Computation took: " << timer.getTimeInMilliseconds() << " ms"
            << std::endl;

  cudaMemcpy(results.get(), dptr_results, fft_length * sizeof(__half2),
             cudaMemcpyDeviceToHost);
  WriteResultsToFileHalf2("cuFFTresults.dat", fft_length, results.get());

  cudaFree(dptr_data);
  cudaFree(dptr_results);
}
