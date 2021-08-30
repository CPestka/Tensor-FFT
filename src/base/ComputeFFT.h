//Contains Functions to compute ffts of half precission data. The neccesary
//cpying operations to and from the GPU are handled by methods of the
//DataHandler class which also holds the device ptr to the according data.
//The parameters of the fft are hold in the struct Plan, which should be
//produced via the function CreatePlan().
//The computation n ffts of a one given length is typicaly performed the
//following way: 1. Create Plan 2. Create DataHandler 3.1. Cpy data to GPU using
//e.g. CopyDataHostToDevice() method of DataHandler 3.2. Call function
//ComputeFFT() 3.3 Cpy results back via method CopyResultsDeviceToHost()
//Repeating step 3. n times.
//Due to the usage of tensor cores the minimal input size is 16^2. All
//other powers of of two are supported as input sizes. Performance is expected
//to be best (compared to other fft libaries) if the input size N is large and
//if N= 16^l * 2^k (while keeping k as small as possible) k is small. This is
//due to the fact that the radix 16 part of the algorithm is accelerated by
//tensor cores compared to the radix 2 part for which this is not the case.
#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "Plan.h"
#include "DataHandler.h"
#include "TensorFFT256.cu"
#include "TensorFFT4096.cu"
#include "TensorRadix16.cu"
#include "Radix2.cu"

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

//Computes a sigle FFT.
//If the GPU isnt satureted with one FFT and there are multiple FFTs to compute
//using the async version below should increase performance.
//Uses default stream -> no cudaDeviceSynchronize() calls between computation
//and memory copies needed unless memcpies are smaller than 64K.
template <typename Integer>
std::optional<std::string> ComputeFFT(Plan<Integer> &fft_plan,
                                      const DataHandler<Integer> &data,
                                      const int max_no_optin_shared_mem =
                                      32768){
  //Use opt in shared memory if required
  if (fft_plan.base_fft_shared_mem_in_bytes_ > max_no_optin_shared_mem) {
    if (fft_plan.base_fft_mode_ == Mode_256) {
      cudaFuncSetAttribute(TensorFFT256<Integer>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           fft_plan.base_fft_shared_mem_in_bytes_);
    } else {
      cudaFuncSetAttribute(TensorFFT4096<Integer>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           fft_plan.base_fft_shared_mem_in_bytes_);
    }
  }

  //Compute base layer FFT
  if (fft_plan.base_fft_mode_ == Mode_256) {
    TensorFFT256<<<fft_plan.base_fft_gridsize_,
                   fft_plan.base_fft_blocksize_,
                   fft_plan.base_fft_shared_mem_in_bytes_>>>(
        data.dptr_input_RE_, data.dptr_input_IM_,
        data.dptr_results_RE_, data.dptr_results_IM_, fft_plan.fft_length_,
        fft_plan.amount_of_r16_steps_, fft_plan.amount_of_r2_steps_);
  } else {
    TensorFFT4096<<<fft_plan.base_fft_gridsize_,
                    fft_plan.base_fft_blocksize_,
                    fft_plan.base_fft_shared_mem_in_bytes_>>>(
        data.dptr_input_RE_, data.dptr_input_IM_,
        data.dptr_results_RE_, data.dptr_results_IM_, fft_plan.fft_length_,
        fft_plan.amount_of_r16_steps_, fft_plan.amount_of_r2_steps_);
  }

  //For each step the input data is the output data of the previous step
  __half* dptr_current_input_RE = data.dptr_results_RE_;
  __half* dptr_current_input_IM = data.dptr_results_IM_;
  __half* dptr_current_results_RE = data.dptr_input_RE_;
  __half* dptr_current_results_IM = data.dptr_input_IM_;

  Integer sub_fft_length = fft_plan.base_fft_mode_ == Mode_256 ? 256 : 4096;

  //Use opt in shared memory if required
  if (fft_plan.r16_shared_mem_in_bytes_ > max_no_optin_shared_mem) {
    cudaFuncSetAttribute(TensorRadix16<Integer>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         fft_plan.r16_shared_mem_in_bytes_);
  }

  //Launch radix16 kernels
  for(int i = fft_plan.base_fft_mode_ == Mode_256 ? 1 : 2;
      i<fft_plan.amount_of_r16_steps_; i++){

    TensorRadix16<<<fft_plan.r16_gridsize_,
                    fft_plan.r16_blocksize_,
                    fft_plan.r16_shared_mem_in_bytes_>>>(
        dptr_current_input_RE, dptr_current_input_IM,
        dptr_current_results_RE, dptr_current_results_IM,
        fft_plan.fft_length_, sub_fft_length);

    //Update sub_fft_length
    sub_fft_length *= 16;

    std::swap(dptr_current_input_RE, dptr_current_results_RE);
    std::swap(dptr_current_input_IM, dptr_current_results_IM);
  }

  //Radix 2 kernels
  for(int i=0; i<fft_plan.amount_of_r2_steps_; i++){
    int amount_of_r2_blocks = sub_fft_length / fft_plan.r2_blocksize_;
    //One radix2 kernel combines 2 subffts -> if there are N sub_ffts launch N/2
    //Kernels
    int remaining_r2_combines =
        ExactPowerOf2<int>(fft_plan.amount_of_r2_steps_ - 1 - i);

    for(int j=0; j<remaining_r2_combines; j++){
      Integer memory_offset = j * 2 * sub_fft_length;
      Radix2Kernel<<<amount_of_r2_blocks, fft_plan.r2_blocksize_>>>(
          dptr_current_input_RE + memory_offset,
          dptr_current_input_IM + memory_offset,
          dptr_current_results_RE + memory_offset,
          dptr_current_results_IM + memory_offset,
          sub_fft_length);
    }

    //Update sub_fft_length
    sub_fft_length *= 2;

    std::swap(dptr_current_input_RE, dptr_current_results_RE);
    std::swap(dptr_current_input_IM, dptr_current_results_IM);
  }

  if (cudaPeekAtLastError() != cudaSuccess){
    return cudaGetErrorString(cudaPeekAtLastError());
  }
  return std::nullopt;
}

//Accepts multiple ffts at a time. For each fft respectively the corresponding
//memcpys and kernels are issued into one stream respectively, which allows work
//for multiple ffts to be executed concurrently if the recources on the device
//are avaiable.
//The memory requiredments for a batch of n ffts is n*requiredment of a the
//singular one.
//No manual cudaDeviceSynchronize() calls are needed before and after this
//function as they are included at the end of this function and DataBatchHandler
//methods.
template <typename Integer>
std::optional<std::string> ComputeFFT(const Plan<Integer> &fft_plan,
                                      const DataBatchHandler<Integer> &data,
                                      const int max_no_optin_shared_mem){
  //Create a stream for each fft
  std::vector<cudaStream_t> streams;
  streams.resize(data.amount_of_ffts_);
  for(int i=0; i<data.amount_of_ffts_; i++){
    if (cudaStreamCreate(&(streams[i])) != cudaSuccess){
       return cudaGetErrorString(cudaPeekAtLastError());
    }
  }

  //Use opt in shared memory if required
  if (fft_plan.base_fft_shared_mem_in_bytes_ > max_no_optin_shared_mem) {
    if (fft_plan.base_fft_mode_ == Mode_256) {
      cudaFuncSetAttribute(TensorFFT256<Integer>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           fft_plan.base_fft_shared_mem_in_bytes_);
    } else {
      cudaFuncSetAttribute(TensorFFT4096<Integer>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           fft_plan.base_fft_shared_mem_in_bytes_);
    }
  }

  //Compute base layer FFT
  for(int i=0; i<data.amount_of_ffts_; i++){
    if (fft_plan.base_fft_mode_ == Mode_256) {
      TensorFFT256<<<fft_plan.base_fft_gridsize_,
                     fft_plan.base_fft_blocksize_,
                     fft_plan.base_fft_shared_mem_in_bytes_,
                     streams[i]>>>(
          data.dptr_input_RE_[i], data.dptr_input_IM_[i],
          data.dptr_results_RE_[i], data.dptr_results_IM_[i], fft_plan.fft_length_,
          fft_plan.amount_of_r16_steps_, fft_plan.amount_of_r2_steps_);
    } else {
      TensorFFT4096<<<fft_plan.base_fft_gridsize_,
                      fft_plan.base_fft_blocksize_,
                      fft_plan.base_fft_shared_mem_in_bytes_,
                      streams[i]>>>(
          data.dptr_input_RE_[i], data.dptr_input_IM_[i],
          data.dptr_results_RE_[i], data.dptr_results_IM_[i],
          fft_plan.fft_length_, fft_plan.amount_of_r16_steps_,
          fft_plan.amount_of_r2_steps_);
    }
  }

  std::vector<__half*> dptr_current_input_RE;
  std::vector<__half*> dptr_current_input_IM;
  std::vector<__half*> dptr_current_results_RE;
  std::vector<__half*> dptr_current_results_IM;
  dptr_current_input_RE.resize(data.amount_of_ffts_, nullptr);
  dptr_current_input_IM.resize(data.amount_of_ffts_, nullptr);
  dptr_current_results_RE.resize(data.amount_of_ffts_, nullptr);
  dptr_current_results_IM.resize(data.amount_of_ffts_, nullptr);

  std::vector<Integer> sub_fft_length;
  sub_fft_length.resize(data.amount_of_ffts_, 16);

  //For each step the input data is the output data of the previous step
  for(int i=0; i<data.amount_of_ffts_; i++){
    dptr_current_input_RE[i] = data.dptr_results_RE_[i];
    dptr_current_input_IM[i] = data.dptr_results_IM_[i];
    dptr_current_results_RE[i] = data.dptr_input_RE_[i];
    dptr_current_results_IM[i] = data.dptr_input_IM_[i];
    sub_fft_length[i] = fft_plan.base_fft_mode_ == Mode_256 ? 256 : 4096;
  }

  //Use opt in shared memory if required
  if (fft_plan.r16_shared_mem_in_bytes_ > max_no_optin_shared_mem) {
    cudaFuncSetAttribute(TensorRadix16<Integer>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         fft_plan.r16_shared_mem_in_bytes_);
  }

  //Launch radix16 kernels
  for(int i=0; i<data.amount_of_ffts_; i++){
    for(int j = fft_plan.base_fft_mode_ == Mode_256 ? 1 : 2;
        j<fft_plan.amount_of_r16_steps_; j++){
      TensorRadix16<<<fft_plan.r16_gridsize_,
                      fft_plan.r16_blocksize_,
                      fft_plan.r16_shared_mem_in_bytes_,
                      streams[i]>>>(
          dptr_current_input_RE[i], dptr_current_input_IM[i],
          dptr_current_results_RE[i], dptr_current_results_IM[i],
          fft_plan.fft_length_, sub_fft_length[i]);

      //Update sub_fft_length
      sub_fft_length[i] *= 16;

      std::swap(dptr_current_input_RE[i], dptr_current_results_RE[i]);
      std::swap(dptr_current_input_IM[i], dptr_current_results_IM[i]);
    }
  }

  for(int i=0; i<data.amount_of_ffts_; i++){
    //Radix 2 kernels
    for(int j=0; j<fft_plan.amount_of_r2_steps_; j++){
      int remaining_r2_combines =
          ExactPowerOf2<int>(fft_plan.amount_of_r2_steps_ - 1 - j);
      int amount_of_r2_blocks = sub_fft_length[i] / fft_plan.r2_blocksize_;

      //One radix2 kernel combines 2 subffts -> if there are still more than 2
      //launch multiple kernels
      for(int k=0; k<remaining_r2_combines; k++){
        Integer memory_offset = k * 2 * sub_fft_length[i];
        Radix2Kernel<<<amount_of_r2_blocks, fft_plan.r2_blocksize_, 0,
                       streams[i]>>>(
            dptr_current_input_RE[i] + memory_offset,
            dptr_current_input_IM[i] + memory_offset,
            dptr_current_results_RE[i] + memory_offset,
            dptr_current_results_IM[i] + memory_offset,
            sub_fft_length[i]);
      }

      //Update sub_fft_length
      sub_fft_length[i] *= 2;

      std::swap(dptr_current_input_RE[i], dptr_current_results_RE[i]);
      std::swap(dptr_current_input_IM[i], dptr_current_results_IM[i]);
    }
  }

  cudaDeviceSynchronize();

  if (cudaPeekAtLastError() != cudaSuccess){
    return cudaGetErrorString(cudaPeekAtLastError());
  }

  return std::nullopt;
}

/*
//Multi GPU variant of ComputeFFT.
//Computes one fft on each device on inputs hold by data.
std::optional<std::string> ComputeFFTMultiGPU(Plan &fft_plan,
                                              DataHandlerMultiGPU &data){
  for(int n=0; n<static_cast<int>(data.device_ids_.size()); n++){
    cudaSetDevice(data.device_ids_[n]);

    //Launch kernel that performs the transposes to prepare the data for the
    //radix steps
    TransposeKernel<<<fft_plan.transposer_amount_of_blocks_,
                      fft_plan.transposer_blocksize_>>>(
        data.dptr_input_RE_[n], data.dptr_input_IM_[n],
        data.dptr_results_RE_[n], data.dptr_results_IM_[n],
        fft_plan.fft_length_, fft_plan.amount_of_r16_steps_,
        fft_plan.amount_of_r2_steps_);

    DFTKernel<<<fft_plan.dft_amount_of_blocks_,
                32 * fft_plan.dft_warps_per_block_>>>(
        data.dptr_results_RE_[n], data.dptr_results_IM_[n],
        data.dptr_input_RE_[n], data.dptr_input_IM_[n],
        data.dptr_dft_matrix_RE_[n], data.dptr_dft_matrix_IM_[n]);

    __half* dptr_current_input_RE;
    __half* dptr_current_input_IM;
    __half* dptr_current_results_RE;
    __half* dptr_current_results_IM;
    int sub_fft_length = 16;

    //Launch radix16 kernels
    for(int i=0; i<fft_plan.amount_of_r16_steps_; i++){
      //For each step the input data is the output data of the previous step
      if ((i % 2) == 0) {
        dptr_current_input_RE = data.dptr_input_RE_[n];
        dptr_current_input_IM = data.dptr_input_IM_[n];
        dptr_current_results_RE = data.dptr_results_RE_[n];
        dptr_current_results_IM = data.dptr_results_IM_[n];
      } else {
        dptr_current_input_RE = data.dptr_results_RE_[n];
        dptr_current_input_IM = data.dptr_results_IM_[n];
        dptr_current_results_RE = data.dptr_input_RE_[n];
        dptr_current_results_IM = data.dptr_input_IM_[n];
      }

      int shared_mem_in_bytes = fft_plan.r16_warps_per_block_ * 16 * 16 *
                                2 * sizeof(__half);

      if (i == 0) {

        cudaFuncSetAttribute(Radix16KernelFirstStep,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem_in_bytes);

        Radix16KernelFirstStep<<<fft_plan.r16_amount_of_blocks_,
                                 32 * fft_plan.r16_warps_per_block_,
                                 shared_mem_in_bytes>>>(
            dptr_current_input_RE, dptr_current_input_IM,
            dptr_current_results_RE, dptr_current_results_IM,
            data.dptr_dft_matrix_RE_[n], data.dptr_dft_matrix_IM_[n]);
      } else {

        cudaFuncSetAttribute(Radix16Kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem_in_bytes);

        Radix16Kernel<<<fft_plan.r16_amount_of_blocks_,
                        32 * fft_plan.r16_warps_per_block_,
                        shared_mem_in_bytes>>>(
            dptr_current_input_RE, dptr_current_input_IM, dptr_current_results_RE,
            dptr_current_results_IM, fft_plan.fft_length_, sub_fft_length);
      }

      //Update sub_fft_length
      sub_fft_length = sub_fft_length * 16;
    }

    //Radix 2 kernels
    for(int i=0; i<fft_plan.amount_of_r2_steps_; i++){
      //For each step the input data is the output data of the previous step
      if (((i + fft_plan.amount_of_r16_steps_) % 2) == 0) {
        dptr_current_input_RE = data.dptr_results_RE_[n];
        dptr_current_input_IM = data.dptr_results_IM_[n];
        dptr_current_results_RE = data.dptr_input_RE_[n];
        dptr_current_results_IM = data.dptr_input_IM_[n];
      } else {
        dptr_current_input_RE = data.dptr_input_RE_[n];
        dptr_current_input_IM = data.dptr_input_IM_[n];
        dptr_current_results_RE = data.dptr_results_RE_[n];
        dptr_current_results_IM = data.dptr_results_IM_[n];
      }

      int remaining_sub_ffts = 1;
      for(int k=0; k<fft_plan.amount_of_r2_steps_ - i; k++){
        remaining_sub_ffts = remaining_sub_ffts * 2;
      }

      int amount_of_r2_blocks = sub_fft_length / fft_plan.r2_blocksize_;

      //One radix2 kernel combines 2 subffts -> if there are still more than 2
      //launch multiple kernels
      for(int j=0; j<(remaining_sub_ffts/2); j++){
        int memory_offset = j * 2 * sub_fft_length;
        Radix2Kernel<<<amount_of_r2_blocks, fft_plan.r2_blocksize_>>>(
            dptr_current_input_RE + memory_offset,
            dptr_current_input_IM + memory_offset,
            dptr_current_results_RE + memory_offset,
            dptr_current_results_IM + memory_offset,
            sub_fft_length);
      }

      //Update sub_fft_length
      sub_fft_length = sub_fft_length * 2;
    }
  }

  return std::nullopt;
}

//Multi GPU variant of ComputeFFTs.
//Computes all ffts on each device respectively on the input hold by data.
std::optional<std::string> ComputeFFTsMultiGPU(Plan &fft_plan,
                                               DataBatchHandlerMultiGPU &data){
  for(int n=0; n<static_cast<int>(data.device_ids_.size()); n++){
    cudaSetDevice(data.device_ids_[n]);

    //Create a stream for each fft
    std::vector<cudaStream_t> streams;
    streams.resize(data.amount_of_ffts_);
    for(int i=0; i<data.amount_of_ffts_; i++){
      if (cudaStreamCreate(&(streams[i])) != cudaSuccess){
         return cudaGetErrorString(cudaPeekAtLastError());
      }
    }

    //Launch kernel that performs the transposes to prepare the data for the
    //radix steps
    for(int i=0; i<data.amount_of_ffts_; i++){
      TransposeKernel<<<fft_plan.transposer_amount_of_blocks_,
                        fft_plan.transposer_blocksize_, 0, streams[i]>>>(
          data.dptr_input_RE_[n][i], data.dptr_input_IM_[n][i],
          data.dptr_results_RE_[n][i], data.dptr_results_IM_[n][i],
          fft_plan.fft_length_, fft_plan.amount_of_r16_steps_,
          fft_plan.amount_of_r2_steps_);
    }

    //Launch baselayer DFT step kernel
    for(int i=0; i<data.amount_of_ffts_; i++){
      DFTKernel<<<fft_plan.dft_amount_of_blocks_,
                  32 * fft_plan.dft_warps_per_block_, 0, streams[i]>>>(
          data.dptr_results_RE_[n][i], data.dptr_results_IM_[n][i],
          data.dptr_input_RE_[n][i], data.dptr_input_IM_[n][i],
          data.dptr_dft_matrix_RE_[n][i], data.dptr_dft_matrix_IM_[n][i]);
    }

    std::vector<__half*> dptr_current_input_RE;
    std::vector<__half*> dptr_current_input_IM;
    std::vector<__half*> dptr_current_results_RE;
    std::vector<__half*> dptr_current_results_IM;
    dptr_current_input_RE.resize(data.amount_of_ffts_, nullptr);
    dptr_current_input_IM.resize(data.amount_of_ffts_, nullptr);
    dptr_current_results_RE.resize(data.amount_of_ffts_, nullptr);
    dptr_current_results_IM.resize(data.amount_of_ffts_, nullptr);

    std::vector<int> sub_fft_length;
    sub_fft_length.resize(data.amount_of_ffts_, 16);

    int shared_mem_in_bytes = fft_plan.r16_warps_per_block_ *
                              16 * 16 * 2 * sizeof(__half);

    cudaFuncSetAttribute(Radix16KernelFirstStep,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_in_bytes);
    cudaFuncSetAttribute(Radix16Kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_in_bytes);


    for(int i=0; i<data.amount_of_ffts_; i++){
      for(int j=0; j<fft_plan.amount_of_r16_steps_; j++){
        //For each step the input data is the output data of the previous step
        if ((j % 2) == 0) {
          dptr_current_input_RE[i] = data.dptr_input_RE_[n][i];
          dptr_current_input_IM[i] = data.dptr_input_IM_[n][i];
          dptr_current_results_RE[i] = data.dptr_results_RE_[n][i];
          dptr_current_results_IM[i] = data.dptr_results_IM_[n][i];
        } else {
          dptr_current_input_RE[i] = data.dptr_results_RE_[n][i];
          dptr_current_input_IM[i] = data.dptr_results_IM_[n][i];
          dptr_current_results_RE[i] = data.dptr_input_RE_[n][i];
          dptr_current_results_IM[i] = data.dptr_input_IM_[n][i];
        }

        if (j == 0) {
          Radix16KernelFirstStep<<<fft_plan.r16_amount_of_blocks_,
                                   32 * fft_plan.r16_warps_per_block_,
                                   shared_mem_in_bytes, streams[i]>>>(
              dptr_current_input_RE[i], dptr_current_input_IM[i],
              dptr_current_results_RE[i], dptr_current_results_IM[i],
              data.dptr_dft_matrix_RE_[n][i], data.dptr_dft_matrix_IM_[n][i]);
        } else {
          Radix16Kernel<<<fft_plan.r16_amount_of_blocks_,
                         32 * fft_plan.r16_warps_per_block_,
                         shared_mem_in_bytes, streams[i]>>>(
              dptr_current_input_RE[i], dptr_current_input_IM[i],
              dptr_current_results_RE[i], dptr_current_results_IM[i],
              fft_plan.fft_length_, sub_fft_length[i]);
        }

        //Update sub_fft_length
        sub_fft_length[i] = sub_fft_length[i] * 16;
      }
    }

    for(int i=0; i<data.amount_of_ffts_; i++){
      //Radix 2 kernels
      for(int j=0; j<fft_plan.amount_of_r2_steps_; j++){
        //For each step the input data is the output data of the previous step
        if (((j + fft_plan.amount_of_r16_steps_) % 2) == 0) {
          dptr_current_input_RE[i] = data.dptr_input_RE_[n][i];
          dptr_current_input_IM[i] = data.dptr_input_IM_[n][i];
          dptr_current_results_RE[i] = data.dptr_results_RE_[n][i];
          dptr_current_results_IM[i] = data.dptr_results_IM_[n][i];
        } else {
          dptr_current_input_RE[i] = data.dptr_results_RE_[n][i];
          dptr_current_input_IM[i] = data.dptr_results_IM_[n][i];
          dptr_current_results_RE[i] = data.dptr_input_RE_[n][i];
          dptr_current_results_IM[i] = data.dptr_input_IM_[n][i];
        }

        int amount_of_r2_blocks = sub_fft_length[i] / fft_plan.r2_blocksize_;

        int remaining_sub_ffts = 1;
        for(int k=0; k<fft_plan.amount_of_r2_steps_ - j; k++){
          remaining_sub_ffts = remaining_sub_ffts * 2;
        }

        //One radix2 kernel combines 2 subffts -> if there are still more than 2
        //launch multiple kernels
        for(int k=0; k<(remaining_sub_ffts/2); k++){
          int memory_offset = k * sub_fft_length[i];
          Radix2Kernel<<<amount_of_r2_blocks, fft_plan.r2_blocksize_,
                         0, streams[i]>>>(
              dptr_current_input_RE[i] + memory_offset,
              dptr_current_input_IM[i] + memory_offset,
              dptr_current_results_RE[i] + memory_offset,
              dptr_current_results_IM [i]+ memory_offset,
              sub_fft_length[i]);
        }

        //Update sub_fft_length
        sub_fft_length[i] = sub_fft_length[i] * 2;
      }
    }
  }

  for(int n=0; n<static_cast<int>(data.device_ids_.size()); n++){
    cudaSetDevice(data.device_ids_[n]);
    cudaDeviceSynchronize();
  }

  return std::nullopt;
}
*/
