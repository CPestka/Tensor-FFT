//Used to test correctness of dft matrix computed on the gpu

#pragma once

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../base/ComputeDFTMatrix.cu"

bool dft_matrix16_test(){
  float dft_matrix_cpu_RE[16][16];
  float dft_matrix_cpu_IM[16][16];

  std::cout << "PI = " << M_PI << std::endl;

  for(int i=0; i<16; i++){
    for(int j=0; j<16; j++){
      dft_matrix_cpu_RE[j][i] = cos((2*M_PI*i*j)/16);
      dft_matrix_cpu_IM[j][i] = -sin((2*M_PI*i*j)/16);
    }
  }

  std::unique_ptr<__half[]> dft_matrix_gpu_RE =
      std::make_unique<__half[]>(16*16);
  std::unique_ptr<__half[]> dft_matrix_gpu_IM =
      std::make_unique<__half[]>(16*16);

  __half* dptr_dft_matrix_gpu_RE;
  __half* dptr_dft_matrix_gpu_IM;
  cudaMalloc((void**)(&dptr_dft_matrix_gpu_RE), sizeof(__half)*16*16);
  cudaMalloc((void**)(&dptr_dft_matrix_gpu_IM), sizeof(__half)*16*16);

  ComputeDFTMatrix<<<1,16>>>(dptr_dft_matrix_gpu_RE, dptr_dft_matrix_gpu_IM);

  cudaMemcpy(dft_matrix_gpu_RE.get(), dptr_dft_matrix_gpu_RE,
             16*16*sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpy(dft_matrix_gpu_IM.get(), dptr_dft_matrix_gpu_IM,
             16*16*sizeof(__half), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for(int j=0; j<16; j++){
    for(int i=0; i<16; i++){
      double gpu_RE = dft_matrix_gpu_RE[i + 16*j];
      double gpu_IM = dft_matrix_gpu_IM[i + 16*j];

      if ((fabs((dft_matrix_cpu_RE[j][i] - gpu_RE)) > 0.0001) ||
          (fabs((dft_matrix_cpu_IM[j][i] - gpu_IM)) > 0.0001)){
        std::cout << "DFT matrix on CPU and GPU are different!"
                  << std::endl;
        return false;
      }
    }
  }

  std::cout << "CPU_RE:" << std::endl;
  for(int j=0; j<16; j++){
    for(int i=0; i<16; i++){
      std::cout << dft_matrix_cpu_RE[j][i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "CPU_IM:" << std::endl;
  for(int j=0; j<16; j++){
    for(int i=0; i<16; i++){
      std::cout << dft_matrix_cpu_IM[j][i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "GPU_RE:" << std::endl;
  for(int j=0; j<16; j++){
    for(int i=0; i<16; i++){
      std::cout << dft_matrix_gpu_RE[i + 16*j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "GPU_IM:" << std::endl;
  for(int j=0; j<16; j++){
    for(int i=0; i<16; i++){
      std::cout << dft_matrix_gpu_IM[i + 16*j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return true;
}
