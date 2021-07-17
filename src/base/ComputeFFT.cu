//Contains Functions to compute ffts of half precission data. The data is passed
//in form of one ptr which is 2*size of fft long and holds the RE in the first
//and IM in the second part. The parameters of the fft are hold in the struct
//Plan which should be produced via the function CreatePlan(). The results are
//returned in place of the input data.
//Due to the usage of tensor cores the minimal input size is 16^2. All
//other powers of of two are supported as input sizes. Performance is expected
//to be best (compared to other fft libaries) if the input size N is large and
//if N= 16^l * 2^k (while keeping k as small as possible) k is small. This is
//due to the fact that the radix 16 part of the algorithm is accelerated by
//tensor cores compared to the radix 2 part for which this is not the case.
#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <optional>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ComputeDFTMatrix.cu"
#include "Transposer.cu"
#include "TensorDFT16.cu"
#include "TensorRadix16.cu"
#include "Radix2.cu"

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
    return cudaGetErrorString(cudaPeekAtLastError());
  }

  std::optional<std::string> CopyDataFromHostToDevice(__half* data) {
    if (cudaMemcpy(dptr_input_RE_, data, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyHostToDevice)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyResultsFromDeviceToHost(__half* data) {
    if (cudaMemcpy(data, dptr_results_RE_, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyHostToDevice)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyDataFromHostToDeviceAsync(
      __half* data, cudaStream_t &stream) {
    if (cudaMemcpyAsync(dptr_input_RE_, data, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyHostToDevice, stream)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyResultsFromDeviceToHostAsync(
      __half* data, cudaStream_t &stream) {
    if (cudaMemcpyAsync(data, dptr_results_RE_,
                        2 * fft_length_ * sizeof(__half),
                        cudaMemcpyHostToDevice, stream)
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

std::optional<std::string> ComputeFFT(Plan &fft_plan, DataHandler &data){
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  TransposeKernel<<<fft_plan.transposer_amount_of_blocks_,
                    fft_plan.transposer_blocksize_>>>(
      data.dptr_input_RE_, data.dptr_input_IM_, data.dptr_results_RE_,
      data.dptr_results_IM_, fft_plan.fft_length_,
      fft_plan.amount_of_r16_steps_, fft_plan.amount_of_r2_steps_);

  //Launch baselayer DFT step kernel
  DFTKernel<<<fft_plan.dft_amount_of_blocks_,
              32 * fft_plan.dft_warps_per_block_>>>(
      data.dptr_input_RE, data.dptr_input_IM, data.dptr_results_RE,
      data.dptr_results_IM, data.dptr_dft_matrix_RE_, data.dptr_dft_matrix_IM_);

  __half* dptr_current_input_RE;
  __half* dptr_current_input_IM;
  __half* dptr_current_results_RE;
  __half* dptr_current_results_IM;
  int sub_fft_length = 16;

  //Launch radix16 kernels
  for(int i=0; i<fft_plan.amount_of_r16_steps_; i++){
    //For each step the input data is the output data of the previous step
    if ((i % 2) == 0) {
      dptr_current_input_RE = data.dptr_results_RE;
      dptr_current_input_IM = data.dptr_results_IM;
      dptr_current_results_RE = data.dptr_input_RE;
      dptr_current_results_IM = data.dptr_input_IM;
    } else {
      dptr_current_input_RE = data.dptr_input_RE;
      dptr_current_input_IM = data.dptr_input_IM;
      dptr_current_results_RE = data.dptr_results_RE;
      dptr_current_results_IM = data.dptr_results_IM;
    }

    int shared_mem_in_bytes = fft_plan.r16_warps_per_block_ * 16 * 16 *
                              2 * sizeof(__half);
    cudaFuncSetAttribute(Radix16Kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_in_bytes);
    Radix16Kernel<<<fft_plan.r16_amount_of_blocks_,
                    32 * fft_plan.r16_warps_per_block_, shared_mem_in_bytes>>>(
        dptr_current_input_RE, dptr_current_input_IM, dptr_current_results_RE,
        dptr_current_results_IM, data.dptr_dft_matrix_RE_,
        data.dptr_dft_matrix_IM_, fft_plan.fft_length_, sub_fft_length, i);

    //Update sub_fft_length
    sub_fft_length = sub_fft_length * 16;
  }

  //Radix 2 kernels
  for(int i=0; i<fft_plan.amount_of_r2_steps_; i++){
    //For each step the input data is the output data of the previous step
    if (((i + fft_plan.amount_of_r16_steps_) % 2) == 0) {
      dptr_current_input_RE = data.dptr_results_RE;
      dptr_current_input_IM = data.dptr_results_IM;
      dptr_current_results_RE = data.dptr_input_RE;
      dptr_current_results_IM = data.dptr_input_IM;
    } else {
      dptr_current_input_RE = data.dptr_input_RE;
      dptr_current_input_IM = data.dptr_input_IM;
      dptr_current_results_RE = data.dptr_results_RE;
      dptr_current_results_IM = data.dptr_results_IM;
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

  return std::nullopt;
}



//Similar to ComputeFFT() but accepts multiple ffts at a time. For each fft
//respectively the corresponding memcpys and kernels are issued into one stream
//respectively, which allows work for multiple ffts to be executed concurrently
//if the recources on the device are avaiable.
//The memory requiredments are the same as for ComputeFFT() but added together
//for each fft.
std::optional<std::string> ComputeFFTs(std::vector<Plan> &fft_plans,
                                       std::vector<DataHandler> &data,
                                       std::vector<cudaStream_t> &streams){
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    TransposeKernel<<<fft_plans[i].transposer_amount_of_blocks_,
                      fft_plans[i].transposer_blocksize_, 0, streams[i]>>>(
        data[i].dptr_input_RE, data[i].dptr_input_IM, data[i].dptr_results_RE,
        data[i].dptr_results_IM, fft_plans[i].fft_length_,
        fft_plans[i].amount_of_r16_steps_, fft_plans[i].amount_of_r2_steps_);
  }

  //Launch baselayer DFT step kernel
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    DFTKernel<<<fft_plans[i].dft_amount_of_blocks_,
                32 * fft_plans[i].dft_warps_per_block_, 0, streams[i]>>>(
        data[i].dptr_input_RE, data[i].dptr_input_IM, data[i].dptr_results_RE,
        data[i].dptr_results_IM, data[i].dptr_dft_matrix_RE_,
        data[i].dptr_dft_matrix_IM_);
  }

  std::vector<__half*> dptr_current_input_RE;
  std::vector<__half*> dptr_current_input_IM;
  std::vector<__half*> dptr_current_results_RE;
  std::vector<__half*> dptr_current_results_IM;
  std::vector<int> sub_fft_length;
  dptr_current_input_RE.resize(fft_plans.size(), nullptr);
  dptr_current_input_IM.resize(fft_plans.size(), nullptr);
  dptr_current_results_RE.resize(fft_plans.size(), nullptr);
  dptr_current_results_IM.resize(fft_plans.size(), nullptr);
  sub_fft_length.resize(fft_plans.size(), 16);

  std::vector<int> shared_mem_in_bytes;

  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    for(int j=0; j<fft_plans[i].amount_of_r16_steps_; j++){
      //For each step the input data is the output data of the previous step
      if ((j % 2) == 0) {
        dptr_current_input_RE[i] = data[i].dptr_results_RE;
        dptr_current_input_IM[i] = data[i].dptr_results_IM;
        dptr_current_results_RE[i] = data[i].dptr_input_RE;
        dptr_current_results_IM[i] = data[i].dptr_input_IM;
      } else {
        dptr_current_input_RE[i] = data[i].dptr_input_RE;
        dptr_current_input_IM[i] = data[i].dptr_input_IM;
        dptr_current_results_RE[i] = data[i].dptr_results_RE;
        dptr_current_results_IM[i] = data[i].dptr_results_IM;
      }

      shared_mem_in_bytes.push_back(fft_plans[i].r16_warps_per_block_ *
                                    16 * 16 * 2 * sizeof(__half));
      cudaFuncSetAttribute(Radix16Kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shared_mem_in_bytes[i]);

      Radix16Kernel<<<fft_plans[i].amount_of_r16_blocks,
                     32 * fft_plans[i].r16_warps_per_block_,
                     shared_mem_in_bytes[i], streams[i]>>>(
          dptr_current_input_RE[i], dptr_current_input_IM[i],
          dptr_current_results_RE[i], dptr_current_results_IM[i],
          data[i].dptr_dft_matrix_RE_, data[i].dptr_dft_matrix_IM_,
          fft_plans[i].fft_length_, sub_fft_length[i], j);

      //Update sub_fft_length
      sub_fft_length[i] = sub_fft_length[i] * 16;
    }
  }

  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    //Radix 2 kernels
    for(int j=0; j<fft_plans[i].amount_of_r2_steps_; j++){
      //For each step the input data is the output data of the previous step
      if (((j + fft_plans[i].amount_of_r16_steps_) % 2) == 0) {
        dptr_current_input_RE[i] = data[i].dptr_results_RE;
        dptr_current_input_IM[i] = data[i].dptr_results_IM;
        dptr_current_results_RE[i] = data[i].dptr_input_RE;
        dptr_current_results_IM[i] = data[i].dptr_input_IM;
      } else {
        dptr_current_input_RE[i] = data[i].dptr_input_RE;
        dptr_current_input_IM[i] = data[i].dptr_input_IM;
        dptr_current_results_RE[i] = data[i].dptr_results_RE;
        dptr_current_results_IM[i] = data[i].dptr_results_IM;
      }

      int amount_of_r2_blocks = sub_fft_length / fft_plans[i].r2_blocksize_;

      int remaining_sub_ffts = 1;
      for(int k=0; k<fft_plans[i].amount_of_r2_steps_ - j; k++){
        remaining_sub_ffts = remaining_sub_ffts * 2;
      }

      //One radix2 kernel combines 2 subffts -> if there are still more than 2
      //launch multiple kernels
      for(int k=0; k<(remaining_sub_ffts/2); k++){
        int memory_offset = k * sub_fft_length[i];
        Radix2Kernel<<<amount_of_r2_blocks, fft_plans[i].r2_blocksize_,
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

  return std::nullopt;
}

//Create a stream for each fft
std::vector<cudaStream_t> streams;
streams.resize(fft_plans.size());
for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
  if (cudaStreamCreate(&(streams[i])) != cudaSuccess){
     return cudaGetErrorString(cudaPeekAtLastError());
  }
}
