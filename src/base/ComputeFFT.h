//Contains Functions to compute ffts of half precission data. The neccesary
//memory allocation and cpying are left to the user to be performed explicitly.
//Two memory buffers that are large enoguh to hold the data are required on the
//GPU. In which of those buffers the result will be located can be determined by
//the value of the plan results_in_results_ after calling ConfigurePlan().
//The parameters of the fft are hold in the struct Plan, which should be
//produced via the function CreatePlan().
//The computation n ffts of a one given length is typicaly performed the
//following way:
//1.   Create Plan and allocate memory on GPU
//2.1. Cpy data to GPU
//3.2. Call function ComputeFFT()
//3.3  Cpy results back via method CopyResultsDeviceToHost()
//4.   Repeating step 3. n times.
//The minimal input size is 16^3. All other powers of of two are supported as
//input sizes. Performance is expected to be best (compared to other fft
//libaries) if the input size N is large and if N= 16^l * 2^k (while keeping
//k as small as possible) k is small. This is due to the fact that the radix
//16 part of the algorithm is accelerated by tensor cores compared to the radix
//2 part for which this is not the case.
#pragma once

#include <vector>
#include <optional>
#include <string>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "Plan.h"
#include "Transpose.cu"
#include "TensorFFT4KBase.cu"
#include "TensorFFTR16.cu"
#include "FFTR2.cu"

//Computes a sigle FFT.
//Uses default stream -> no cudaDeviceSynchronize() calls between computation
//and memory copies needed unless memcpies are smaller than 64K.
//max_no_optin_shared_mem can be determined via GetMaxNoOptInSharedMem() for a
//given device
//Use Integer = int for fft_length < 2**32 and long long else
template <typename Integer>
std::optional<std::string> ComputeFFT(Plan &fft_plan,
                                      __half2* dptr_input_data,
                                      __half2* dptr_output_data,
                                      int max_no_optin_shared_mem =
                                      32768){
  //Above a certain threshold, if more shared memory is requested it has to be
  //manualy enabled with the specific threshold depending on the used GPU.
  if (fft_plan.transpose_config_.shared_mem_in_bytes_ >
      max_no_optin_shared_mem){
    cudaFuncSetAttribute(Transposer<Integer>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         fft_plan.transpose_config_.shared_mem_in_bytes_);
  }
  if (fft_plan.base_fft_config_.shared_mem_in_bytes_ >
      max_no_optin_shared_mem){
    cudaFuncSetAttribute(TensorFFT4096<Integer>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         fft_plan.base_fft_config_.shared_mem_in_bytes_);
  }
  if (fft_plan.r16_config_.shared_mem_in_bytes_ >
      max_no_optin_shared_mem){
    cudaFuncSetAttribute(TensorRadix16<Integer>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         fft_plan.r16_config_.shared_mem_in_bytes_);
  }

  Transposer<Integer><<<fft_plan.transpose_config_.gridsize_,
                        fft_plan.transpose_config_.blocksize_,
                        fft_plan.transpose_config_.shared_mem_in_bytes_>>>(
      dptr_input_data,
      dptr_output_data,
      static_cast<Integer>(fft_plan.fft_length_),
      fft_plan.amount_of_r16_steps_,
      fft_plan.amount_of_r2_steps_);

  if (cudaPeekAtLastError() != cudaSuccess){
    return cudaGetErrorString(cudaPeekAtLastError());
  }

  //For each step the input data is the output data of the previous step
  __half2* dptr_current_input_data = dptr_output_data;
  __half2* dptr_current_output_data = dptr_input_data;

  TensorFFT4096<Integer><<<fft_plan.base_fft_config_.gridsize_,
                           fft_plan.base_fft_config_.blocksize_,
                           fft_plan.base_fft_config_.shared_mem_in_bytes_>>>(
      dptr_current_input_data,
      dptr_current_output_data,
      static_cast<Integer>(fft_plan.fft_length_),
      fft_plan.amount_of_r16_steps_,
      fft_plan.amount_of_r2_steps_);

  if (cudaPeekAtLastError() != cudaSuccess){
    return cudaGetErrorString(cudaPeekAtLastError());
  }

  std::swap(dptr_current_input_data, dptr_current_output_data);

  //Launch radix16 kernels
  for(int i = 0; i<fft_plan.amount_of_r16_kernels_; i++){
    TensorRadix16<Integer><<<fft_plan.r16_config_.gridsize_,
                             fft_plan.r16_config_.blocksize_,
                             fft_plan.r16_config_.shared_mem_in_bytes_>>>(
        dptr_current_input_data,
        dptr_current_output_data,
        static_cast<Integer>(fft_plan.fft_length_),
        fft_plan.sub_fft_length_[i]);

    if (cudaPeekAtLastError() != cudaSuccess){
      return cudaGetErrorString(cudaPeekAtLastError());
    }

    std::swap(dptr_current_input_data, dptr_current_output_data);
  }

  //Radix 2 kernels
  for(int i=0; i<fft_plan.amount_of_r2_steps_; i++){
    Integer current_subfft_length =
        fft_plan.sub_fft_length[fft_plan.amount_of_r16_kernels_ + i];
    int amount_of_r2_blocks = current_subfft_length / fft_plan.r2_blocksize_;

    for(int j=0; j<my_plan.amount_of_r2_kernels_per_r2_step_[i]; j++){
      Integer memory_offset = j * 2 * current_subfft_length;
      Radix2Kernel<<<amount_of_r2_blocks, fft_plan.r2_blocksize_>>>(
          dptr_current_input_data + memory_offset,
          dptr_current_output_data + memory_offset,
          current_subfft_length);
    }

    if (cudaPeekAtLastError() != cudaSuccess){
      return cudaGetErrorString(cudaPeekAtLastError());
    }

    std::swap(dptr_current_input_data, dptr_current_output_data);
  }

  return std::nullopt;
}
