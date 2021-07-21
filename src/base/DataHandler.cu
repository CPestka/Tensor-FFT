#pragma once

#include <iostream>
#include <optional>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ComputeDFTMatrix.cu"

//This calss is used to manage the memory on the device needed for the compution
//of one FFT.
//It is intended to be reused if multiple FFTs are to be performed sequentialy.
//Instantiation results in the allocation of the needed memory and the
//precomputation of the DFT matrices that are needed during the computaion.
//The neccesary memcpys to and from the device before and after the computation
//should be performed via the according methods of this class.
class DataHandler{
public:
  DataHandler(int fft_length) : fft_length_(fft_length) {
    if (cudaMalloc((void**)(&dptr_input_RE_), 4 * sizeof(__half) * fft_length_)
        != cudaSuccess){
       std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    }
    dptr_input_IM_ = dptr_input_RE_ + fft_length_;
    dptr_results_RE_ = dptr_input_IM_ + fft_length_;
    dptr_results_IM_ = dptr_results_RE_ + fft_length_;

    //Here we precompute the dft matrix batches needed for the DFTKernel() and
    //Radix16Kernel(). Currently there is one matrix precomputed for each warp.
    //The other options are to only precompute one (lower memory usage but read
    //conflicts for each warp) and to compute the dft matrix each time during the
    //kernels. (TODO: find out whats "best")
    if (cudaMalloc((void**)(&dptr_dft_matrix_RE_),
                   2 * sizeof(__half) * fft_length_)
        != cudaSuccess) {
      std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    }
    dptr_dft_matrix_IM_ = dptr_dft_matrix_RE_ + fft_length_;

    ComputeDFTMatrix<<<fft_length / 256, 16*16>>>(dptr_dft_matrix_RE_,
                                                  dptr_dft_matrix_IM_);

    cudaDeviceSynchronize();
  }

  std::optional<std::string> PeakAtLastError() {
    if (cudaPeekAtLastError() != cudaSuccess){
      return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyDataHostToDevice(__half* data) {
    if (cudaMemcpy(dptr_input_RE_, data, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyHostToDevice)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyResultsDeviceToHost(__half* data,
                                                     int amount_of_r16_steps,
                                                     int amount_of_r2_steps) {
    __half* results;
    if (((amount_of_r16_steps + amount_of_r2_steps) % 2) != 0) {
      results = dptr_results_RE_;
    } else {
      results = dptr_input_RE_;
    }
    if (cudaMemcpy(data, results, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyDeviceToHost)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyDataHostToDeviceAsync(
      __half* data, cudaStream_t &stream) {
    if (cudaMemcpyAsync(dptr_input_RE_, data, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyHostToDevice, stream)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyResultsDeviceToHostAsync(
      __half* data, int amount_of_r16_steps, int amount_of_r2_steps,
      cudaStream_t &stream) {
    __half* results;
    if (((amount_of_r16_steps + amount_of_r2_steps) % 2) == 1) {
      results = dptr_results_RE_;
    } else {
      results = dptr_input_RE_;
    }
    if (cudaMemcpyAsync(data, results, 2 * fft_length_ * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  ~DataHandler(){
    cudaFree(dptr_dft_matrix_RE_);
    cudaFree(dptr_input_RE_);
  }
  int fft_length_;
  __half* dptr_input_RE_;
  __half* dptr_input_IM_;
  __half* dptr_results_RE_;
  __half* dptr_results_IM_;
  __half* dptr_dft_matrix_RE_;
  __half* dptr_dft_matrix_IM_;
};
