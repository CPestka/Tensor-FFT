//Used to test correctness of the dft kernel
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

#include "TestingDataCreation.cu"
#include "FileWriter.cu"
#include "../base/Transposer.cu"

__global__ void PrepareCuFFTInput(__half* input_RE, __half* input_IM,
                                  __half2* cuFFT_in){
  for(int i=0; i<16; i++){
    cuFFT_in[i] = __halves2half2(input_RE[i], input_IM[i]);
  }
}

__global__ void SaveCuFFTResults(__half2* cuFFT_out, __half* out_RE,
                                 __half* out_IM){
  for(int i=0; i<16; i++){
    out_RE[i] = __low2half(cuFFT_out[i]);
    out_IM[i] = __high2half(cuFFT_out[i]);
  }
}

bool dft16_test(){
  int fft_length = 16*16*16;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data_1 =
      CreateSineSuperpostion(fft_length, weights);
  std::unique_ptr<__half[]> data_2 =
      CreateSineSuperpostion(fft_length, weights);

  WriteResultsToFile("input.dat", fft_length, data_1.get());

  __half* dptr_input_RE;
  __half* dptr_input_IM;
  __half* dptr_results_kernel_RE;
  __half* dptr_results_kernel_IM;
  __half* dptr_results_cuFFT_RE;
  __half* dptr_results_cuFFT_IM;
  cudaMalloc((void**)(&dptr_input_RE), 6 * sizeof(__half) * fft_length);

  dptr_input_IM = dptr_input_RE + fft_length;
  dptr_results_kernel_RE = dptr_input_IM + fft_length;
  dptr_results_kernel_IM = dptr_results_kernel_RE + fft_length;
  dptr_results_cuFFT_RE = dptr_results_kernel_IM + fft_length;
  dptr_results_cuFFT_IM = dptr_results_cuFFT_RE + fft_length;

  cudaMemcpy(dptr_input_RE, data_1.get(), 2 * fft_length * sizeof(__half),
             cudaMemcpyHostToDevice);

  int transpose_blocksize = 256;
  int amount_of_transpose_blocks =
     ceil(static_cast<float>(fft_length) /
          static_cast<float>(transpose_blocksize));

  TransposeKernel<<<amount_of_transpose_blocks, transpose_blocksize>>>(
      dptr_input_RE, dptr_input_IM, dptr_results_kernel_RE,
      dptr_results_kernel_IM, fft_length, 2, 0);

  cudaMemcpy(data_1.get(), dptr_results_RE, 2 * fft_length * sizeof(__half),
             cudaMemcpyDeviceToHost);

  WriteResultsToFile("dft_test_kernel.dat", fft_length, data_1.get());

  __half2* dptr_cuFFT_in;
  __half2* dptr_cuFFT_out;
  cudaMalloc((void**)(&dptr_cuFFT_in), sizeof(__half2) * 16);
  cudaMalloc((void**)(&dptr_cuFFT_out), sizeof(__half2) * 16);

  for(int i=0; i<16*16; i++){
    PrepareCuFFTInput<<<1,1>>>(dptr_input_RE + 16*i, dptr_input_IM + 16*i,
                               dptr_cuFFT_in);

    cufftHandle plan;
    cufftResult r;
    r = cufftCreate(&plan);
    assert(r == CUFFT_SUCCESS);
    size_t size = 0;
    long long fft_length_1 = 16;
    r = cufftXtMakePlanMany(plan, 1, &fft_length_1, nullptr, 1, 1, CUDA_C_16F,
                            nullptr, 1, 1, CUDA_C_16F, 1, &size, CUDA_C_16F);
    assert(r == CUFFT_SUCCESS);
    r = cufftXtExec(plan, dptr_cuFFT_in, dptr_cuFFT_out, CUFFT_FORWARD);
    assert(r == CUFFT_SUCCESS);

    SaveCuFFTResults<<<1,1>>>(dptr_cuFFT_out, dptr_results_cuFFT_RE + 16*i,
                              dptr_results_cuFFT_IM + 16*i);
  }

  cudaMemcpy(data_2.get(), dptr_results_cuFFT_RE,
             2 * fft_length * sizeof(__half), cudaMemcpyDeviceToHost);

  WriteResultsToFile("dft_test_cuFFT.dat", fft_length, data_2.get());

  for(int i=0; i<fft_length; i++){
    float cpu_re = data_2[i];
    float gpu_re = data_1[i];
    float cpu_im = data_2[i + fft_length];
    float gpu_im = data_1[i + fft_length];
    if ((cpu_re != gpu_re) || (cpu_im != gpu_im)){
      std::cout << "Results of dfts are different!"
                << "CuFFT: " << cpu_re << " " << cpu_im << " Kernel: " << gpu_re
                << " " << gpu_im << std::endl;
      if ((fabs(cpu_re - gpu_re) > 0.01) || (fabs(cpu_im - gpu_im) > 0.01)){
        return false;
      }
    }
  }

  cudaFree(dptr_input_RE);
  cudaFree(dptr_cuFFT_in);
  cudaFree(dptr_cuFFT_out);

  return true;
}
