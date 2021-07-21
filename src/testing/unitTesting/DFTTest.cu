//Used to test correctness of the dft kernel

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../TestingDataCreation.cu"
#include "../FileWriter.cu"
#include "../../base/Transposer.cu"
#include "../../base/TensorDFT16.cu"
#include "../../base/ComputeDFTMatrix.cu"

//Tests the dft kernel on zero valued data
bool TestDFTKernel_0(){
  std::unique_ptr<__half[]> data_RE =
      std::make_unique<__half[]>(16*16*16);
  std::unique_ptr<__half[]> data_IM =
      std::make_unique<__half[]>(16*16*16);

  //Set input data
  for(int i=0; i<16*16*16; i++){
    data_RE[i] = 0;
    data_IM[i] = 0;
  }

  __half* dptr_data_RE;
  __half* dptr_data_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  cudaMalloc((void**)(&dptr_data_RE), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_data_IM), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_results_RE), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_results_IM), sizeof(__half)*16*16*16);

  cudaMemcpy(dptr_data_RE, data_RE.get(), 16*16*16*sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dptr_data_IM, data_IM.get(), 16*16*16*sizeof(__half),
             cudaMemcpyHostToDevice);

  __half* dptr_dft_matrix_RE;
  __half* dptr_dft_matrix_IM;
  cudaMalloc((void**)(&dptr_dft_matrix_RE), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_dft_matrix_IM), sizeof(__half)*16*16*16);

  //Compute the neccesary dft matrices
  ComputeDFTMatrix<<<16,16*16>>>(dptr_dft_matrix_RE, dptr_dft_matrix_IM);

  //Computation of dft
  DFTKernel<<<4,4*32>>>(dptr_data_RE, dptr_data_IM, dptr_results_RE,
                        dptr_results_IM, dptr_dft_matrix_RE,
                        dptr_dft_matrix_IM);

  cudaMemcpy(data_RE.get(), dptr_results_RE, 16*16*16*sizeof(__half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data_IM.get(), dptr_results_IM, 16*16*16*sizeof(__half),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  std::cout << WriteResultsToFile(
      "test_dft_0.dat", 16*16*16, data_RE.get(), data_IM.get()).value_or("")
            << std::endl;

  cudaFree(dptr_dft_matrix_IM);
  cudaFree(dptr_dft_matrix_RE);
  cudaFree(dptr_results_IM);
  cudaFree(dptr_results_RE);
  cudaFree(dptr_data_IM);
  cudaFree(dptr_data_RE);

  //Results should all be zero valued as well
  for(int i=0; i<16*16; i++){
    for(int j=0; j<16; j++){
      bool RE_correct = fabs(static_cast<double>(data_RE[16*i + j])) < 0.002;
      bool IM_correct = fabs(static_cast<double>(data_IM[16*i + j])) < 0.002;
      if ((!RE_correct) || (!IM_correct)) {
        std::cout << "Results of dft by Kernel are incorrect for example data "
                  << "of only 0." << std::endl;
        return false;
      }
    }
  }

  return true;
}

//Test dft kernel on simple sine wave
bool TestDFTKernelSin_16(){
  std::unique_ptr<__half[]> data_RE =
      std::make_unique<__half[]>(16*16*16);
  std::unique_ptr<__half[]> data_IM =
      std::make_unique<__half[]>(16*16*16);

  //Set input data
  for(int j=0; j<16*16; j++){
    for(int i=0; i<16; i++){
      data_RE[j*16 + i] = sin((2*M_PI*i)/16.0);
      data_IM[j*16 + i] = 0;
    }
  }

  __half* dptr_data_RE;
  __half* dptr_data_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  cudaMalloc((void**)(&dptr_data_RE), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_data_IM), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_results_RE), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_results_IM), sizeof(__half)*16*16*16);

  cudaMemcpy(dptr_data_RE, data_RE.get(), 16*16*16*sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dptr_data_IM, data_IM.get(), 16*16*16*sizeof(__half),
             cudaMemcpyHostToDevice);

  __half* dptr_dft_matrix_RE;
  __half* dptr_dft_matrix_IM;
  cudaMalloc((void**)(&dptr_dft_matrix_RE), sizeof(__half)*16*16*16);
  cudaMalloc((void**)(&dptr_dft_matrix_IM), sizeof(__half)*16*16*16);

  //Compute the neccesary dft matrices
  ComputeDFTMatrix<<<16,16*16>>>(dptr_dft_matrix_RE, dptr_dft_matrix_IM);

  //Computation of dft
  DFTKernel<<<4,4*32>>>(dptr_data_RE, dptr_data_IM, dptr_results_RE,
                       dptr_results_IM, dptr_dft_matrix_RE, dptr_dft_matrix_IM);

  cudaMemcpy(data_RE.get(), dptr_results_RE, 16*16*16*sizeof(__half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data_IM.get(), dptr_results_IM, 16*16*16*sizeof(__half),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  std::cout << WriteResultsToFile(
      "test_dft_sin_16.dat", 16*16*16, data_RE.get(),
      data_IM.get()).value_or("")
            << std::endl;

  cudaFree(dptr_dft_matrix_IM);
  cudaFree(dptr_dft_matrix_RE);
  cudaFree(dptr_results_IM);
  cudaFree(dptr_results_RE);
  cudaFree(dptr_data_IM);
  cudaFree(dptr_data_RE);

  //Compare results to known actual result
  for(int i=0; i<16*16; i++){
    for(int j=0; j<16; j++){
      bool RE_correct = fabs(static_cast<double>(data_RE[16*i + j])) < 0.01;

      bool IM_correct;
      if (j==1 || j==15){
        if (j==1){
          IM_correct = fabs(static_cast<double>(data_IM[16*i + j]) + 8.0) < 0.01;
        }
        if (j==15) {
          IM_correct = fabs(static_cast<double>(data_IM[16*i + j]) - 8.0) < 0.01;
        }
      } else {
        IM_correct = fabs(static_cast<double>(data_IM[16*i + j])) < 0.01;
      }

      if ((!RE_correct) || (!IM_correct)) {
        std::cout << "Results of dft by Kernel are incorrect for example data "
                  << "of sin(x) x e [0:2*PI]." << std::endl;
        std::cout << "i= " << i << " j= " << j
                  << " RE= " << static_cast<double>(data_RE[16*i + j])
                  << " IM= " << static_cast<double>(data_IM[16*i + j])
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}

//Test dft kernel on simple sine wave
bool TestDFTKernelSin_2(){
  std::unique_ptr<__half[]> data_RE =
      std::make_unique<__half[]>(16*16*16*16*2);
  std::unique_ptr<__half[]> data_IM =
      std::make_unique<__half[]>(16*16*16*16*2);

  //Set input data
  for(int j=0; j<16*16*16*2; j++){
    for(int i=0; i<16; i++){
      data_RE[j*16 + i] = sin((2*M_PI*i)/16.0);
      data_IM[j*16 + i] = 0;
    }
  }

  __half* dptr_data_RE;
  __half* dptr_data_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  cudaMalloc((void**)(&dptr_data_RE), sizeof(__half)*16*16*16*16*2);
  cudaMalloc((void**)(&dptr_data_IM), sizeof(__half)*16*16*16*16*2);
  cudaMalloc((void**)(&dptr_results_RE), sizeof(__half)*16*16*16*16*2);
  cudaMalloc((void**)(&dptr_results_IM), sizeof(__half)*16*16*16*16*2);

  cudaMemcpy(dptr_data_RE, data_RE.get(), 16*16*16*16*2*sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dptr_data_IM, data_IM.get(), 16*16*16*16*2*sizeof(__half),
             cudaMemcpyHostToDevice);

  __half* dptr_dft_matrix_RE;
  __half* dptr_dft_matrix_IM;
  cudaMalloc((void**)(&dptr_dft_matrix_RE), sizeof(__half)*16*16*16*16*2);
  cudaMalloc((void**)(&dptr_dft_matrix_IM), sizeof(__half)*16*16*16*16*2);

  //Compute the neccesary dft matrices
  ComputeDFTMatrix<<<16*16*2,16*16>>>(dptr_dft_matrix_RE, dptr_dft_matrix_IM);

  //Computation of dft
  DFTKernel<<<4*16*2,4*32>>>(dptr_data_RE, dptr_data_IM, dptr_results_RE,
                             dptr_results_IM, dptr_dft_matrix_RE,
                             dptr_dft_matrix_IM);

  cudaMemcpy(data_RE.get(), dptr_results_RE, 16*16*16*16*2*sizeof(__half),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data_IM.get(), dptr_results_IM, 16*16*16*16*2*sizeof(__half),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  std::cout << WriteResultsToFile(
      "test_dft_sin_2.dat", 16*16*16*16*2, data_RE.get(),
      data_IM.get()).value_or("")
            << std::endl;

  cudaFree(dptr_dft_matrix_IM);
  cudaFree(dptr_dft_matrix_RE);
  cudaFree(dptr_results_IM);
  cudaFree(dptr_results_RE);
  cudaFree(dptr_data_IM);
  cudaFree(dptr_data_RE);

  //Compare results to known actual result
  for(int i=0; i<16*16*16*2; i++){
    for(int j=0; j<16; j++){
      bool RE_correct = fabs(static_cast<double>(data_RE[16*i + j])) < 0.01;

      bool IM_correct;
      if (j==1 || j==15){
        if (j==1){
          IM_correct = fabs(static_cast<double>(data_IM[16*i + j]) + 8.0) < 0.01;
        }
        if (j==15) {
          IM_correct = fabs(static_cast<double>(data_IM[16*i + j]) - 8.0) < 0.01;
        }
      } else {
        IM_correct = fabs(static_cast<double>(data_IM[16*i + j])) < 0.01;
      }

      if ((!RE_correct) || (!IM_correct)) {
        std::cout << "Results of dft by Kernel are incorrect for example data "
                  << "of sin(x) x e [0:2*PI]." << std::endl;
        std::cout << "i= " << i << " j= " << j
                  << " RE= " << static_cast<double>(data_RE[16*i + j])
                  << " IM= " << static_cast<double>(data_IM[16*i + j])
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}
