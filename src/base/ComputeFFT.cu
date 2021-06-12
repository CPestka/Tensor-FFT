//Contains Functions to compute ffts of half precission data. The data is passed
//in form of one ptr which is 2*size of fft long and holds the RE in the first
//and IM in the second part. The parameters of the fft are hold in the struct
//Plan which should be produced via the function CreatePlan(). The results are
//returned in place of the input data.
//Due to API constrains of the tensor cores the minimal input size is 16^3. All
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

//Performs the FFT according to the parameters supplied by fft_plan on the data.
//data is assumed to have 2*fft_length elements, to be contious in memory and
//having all data_RE elements in the first half of the array and the data_IM in
//the second half.
//The roughly required memory on the device is (size_of(__half)=2)*
//(RE and IM -> 2)*(in data + results data + dft matrices)*size_of_fft bytes.
//The factor 3 could be reduced to 2 if the dft matrix is computed on the fly
//instead.
std::optional<std::string> ComputeFFT(Plan fft_plan, __half* data){
  //Allocation of in/out data arrays on the device
  __half* dptr_input_RE;
  __half* dptr_input_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  if (cudaMalloc((void**)(&dptr_input_RE),
                 4 * sizeof(__half) * fft_plan.fft_length_)
      != cudaSuccess){
     return cudaGetErrorString(cudaPeekAtLastError());
  }
  dptr_input_IM = dptr_input_RE + fft_plan.fft_length_;
  dptr_results_RE = dptr_input_IM + fft_plan.fft_length_;
  dptr_results_IM = dptr_results_RE + fft_plan.fft_length_;

  //Memcpy of input data to device
  if (cudaMemcpy(dptr_input_RE, data,
                 2 * fft_plan.fft_length_ * sizeof(__half),
                 cudaMemcpyHostToDevice)
       != cudaSuccess) {
     return cudaGetErrorString(cudaPeekAtLastError());
  }

  int max_amount_of_warps = fft_plan.fft_length_ / (16 * 16 * 16);
  //Here we precompute the dft matrix batches needed for the DFTKernel() and
  //Radix16Kernel(). Currently there is one batch precomputed for each warp. The
  //other options are to only precompute one (lower memory usage but read
  //conflicts for each warp) and to compute the dft matrix each time during the
  //kernels. (TODO: find out whats "best")
  __half* dptr_dft_matrix_batch_RE_;
  __half* dptr_dft_matrix_batch_IM_;
  if (cudaMalloc((void**)(&dptr_dft_matrix_batch_RE_),
                 2 * sizeof(__half) * 16 * 16 * 16 * max_amount_of_warps)
      != cudaSuccess) {
    return cudaGetErrorString(cudaPeekAtLastError());
  }
  dptr_dft_matrix_batch_IM_ =
      dptr_dft_matrix_batch_RE_ + (16 * 16 * 16 * max_amount_of_warps);

  ComputeDFTMatrix<<<max_amount_of_warps*16, 16*16>>>(dptr_dft_matrix_batch_RE_,
                                                     dptr_dft_matrix_batch_IM_);

  int amount_of_transpose_blocks =
     ceil(static_cast<float>(fft_plan.fft_length_) /
          static_cast<float>(fft_plan.transposer_blocksize_));
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  TransposeKernel<<<amount_of_transpose_blocks,
                    fft_plan.transposer_blocksize_>>>(
      dptr_input_RE, dptr_input_IM, dptr_results_RE, dptr_results_IM,
      fft_plan.fft_length_, fft_plan.amount_of_r16_steps_,
      fft_plan.amount_of_r2_steps_);

  int amount_of_dft_blocks =
      fft_plan.fft_length_ / (16 * 16 * 16 * fft_plan.dft_warps_per_block_);
  //Launch baselayer DFT step kernel
  DFTKernel<<<amount_of_dft_blocks, 32 * fft_plan.dft_warps_per_block_>>>(
      dptr_input_RE, dptr_input_IM, dptr_results_RE, dptr_results_IM,
      dptr_dft_matrix_batch_RE_, dptr_dft_matrix_batch_IM_);

  __half* dptr_current_input_RE;
  __half* dptr_current_input_IM;
  __half* dptr_current_results_RE;
  __half* dptr_current_results_IM;
  int sub_fft_length = 16;

  //Launch radix16 kernels
  for(int i=0; i<fft_plan.amount_of_r16_steps_; i++){
    //For each step the input data is the output data of the previous step
    if ((i % 2) == 0) {
      dptr_current_input_RE = dptr_results_RE;
      dptr_current_input_IM = dptr_results_IM;
      dptr_current_results_RE = dptr_input_RE;
      dptr_current_results_IM = dptr_input_IM;
    } else {
      dptr_current_input_RE = dptr_input_RE;
      dptr_current_input_IM = dptr_input_IM;
      dptr_current_results_RE = dptr_results_RE;
      dptr_current_results_IM = dptr_results_IM;
    }

    int amount_of_r16_blocks =
        fft_plan.fft_length_ / (16 * 16 * 16 * fft_plan.r16_warps_per_block_);

    int shared_mem_in_bytes = fft_plan.r16_warps_per_block_ * 16 * 16 * 16 *
                              2 * sizeof(__half);
    cudaFuncSetAttribute(Radix16Kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_in_bytes);
    Radix16Kernel<<<amount_of_r16_blocks, 32 * fft_plan.r16_warps_per_block_,
                    shared_mem_in_bytes>>>(
        dptr_current_input_RE, dptr_current_input_IM, dptr_current_results_RE,
        dptr_current_results_IM, dptr_dft_matrix_batch_RE_,
        dptr_dft_matrix_batch_IM_, fft_plan.fft_length_, sub_fft_length, i);

    //Update sub_fft_length
    sub_fft_length = sub_fft_length * 16;
  }

  //Radix 2 kernels
  for(int i=0; i<fft_plan.amount_of_r2_steps_; i++){
    //For each step the input data is the output data of the previous step
    if (((i + fft_plan.amount_of_r16_steps_) % 2) == 0) {
      dptr_current_input_RE = dptr_results_RE;
      dptr_current_input_IM = dptr_results_IM;
      dptr_current_results_RE = dptr_input_RE;
      dptr_current_results_IM = dptr_input_IM;
    } else {
      dptr_current_input_RE = dptr_input_RE;
      dptr_current_input_IM = dptr_input_IM;
      dptr_current_results_RE = dptr_results_RE;
      dptr_current_results_IM = dptr_results_IM;
    }

    int amount_of_r2_blocks =
        fft_plan.fft_length_ / fft_plan.r2_blocksize_;

    int remaining_sub_ffts = 1;
    for(int k=0; k<fft_plan.amount_of_r2_steps_ - i; k++){
      remaining_sub_ffts = remaining_sub_ffts * 2;
    }

    //One radix2 kernel combines 2 subffts -> if there are still more than 2
    //launch multiple kernels
    for(int j=0; j<(remaining_sub_ffts/2); j++){
      int memory_offset = j * 2 * sub_fft_length;
      std::cout << amount_of_r2_blocks << " " << fft_plan.r2_blocksize_
                << " " << sub_fft_length << std::endl;
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

  //Cpy results back to host
  if (cudaMemcpy(data, dptr_current_results_RE,
                  2 * fft_plan.fft_length_ * sizeof(__half),
                  cudaMemcpyDeviceToHost)
       != cudaSuccess) {
     return cudaGetErrorString(cudaPeekAtLastError());
  }

  //Free device memory
  if (cudaFree(dptr_input_RE) != cudaSuccess) {
     return cudaGetErrorString(cudaPeekAtLastError());
  }
  if (cudaFree(dptr_dft_matrix_batch_RE_) != cudaSuccess) {
     return cudaGetErrorString(cudaPeekAtLastError());
  }

  return std::nullopt;
}

//Similar to ComputeFFT() but accepts multiple ffts at a time. For each fft
//respectively the corresponding memcpys and kernels are issued into one stream
//respectively, which allows work for multiple ffts to be executed concurrently
//if the recources on the device are avaiable.
//The memory requiredments are the same as for ComputeFFT() but added together
//for each fft.
std::optional<std::string> ComputeFFTs(std::vector<Plan> fft_plans,
                                       std::vector<__half*> data){
  //Create a stream for each fft
  std::vector<cudaStream_t> streams;
  streams.resize(fft_plans.size());
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    if (cudaStreamCreate(&(streams[i])) != cudaSuccess){
       return cudaGetErrorString(cudaPeekAtLastError());
    }
  }

  //Allocation of in/out data arrays on the device
  std::vector<__half*> dptr_input_RE;
  std::vector<__half*> dptr_input_IM;
  std::vector<__half*> dptr_results_RE;
  std::vector<__half*> dptr_results_IM;

  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    dptr_input_RE.push_back(nullptr);
    dptr_input_IM.push_back(nullptr);
    dptr_results_RE.push_back(nullptr);
    dptr_results_IM.push_back(nullptr);

    if (cudaMalloc((void**)(&(dptr_input_RE[i])),
                    4 * sizeof(__half) * fft_plans[i].fft_length_)
        != cudaSuccess){
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    dptr_input_IM[i] = dptr_input_RE[i] + fft_plans[i].fft_length_;
    dptr_results_RE[i] = dptr_input_IM[i] + fft_plans[i].fft_length_;
    dptr_results_IM[i] = dptr_results_RE[i] + fft_plans[i].fft_length_;

    //Memcpy of input data to device
    if (cudaMemcpyAsync(dptr_input_RE[i], data[i],
                        2 * fft_plans[i].fft_length_ * sizeof(__half),
                        cudaMemcpyHostToDevice, streams[i])
       != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
  }

  std::vector<int> max_amount_of_warps;
  for(int i=0; fft_plans.size(); i++){
    max_amount_of_warps.push_back(fft_plans[i].fft_length_ / (16 * 16 * 16));
  }

  //Here we precompute the dft matrix batches needed for the DFTKernel() and
  //Radix16Kernel(). Currently there is one batch precomputed for each warp. The
  //other options are to only precompute one (lower memory usage but read
  //conflicts for each warp) and to compute the dft matrix each time during the
  //kernels. (TODO: find out whats "best")
  std::vector<__half*> dptr_dft_matrix_batch_RE_;
  std::vector<__half*> dptr_dft_matrix_batch_IM_;
  dptr_dft_matrix_batch_RE_.resize(fft_plans.size(), nullptr);
  dptr_dft_matrix_batch_IM_.resize(fft_plans.size(), nullptr);
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    if (cudaMalloc((void**)(&(dptr_dft_matrix_batch_RE_[i])),
                   2 * sizeof(__half) * 16 * 16 * 16 * max_amount_of_warps[i])
        != cudaSuccess) {
      return cudaGetErrorString(cudaPeekAtLastError());
    }

    dptr_dft_matrix_batch_IM_[i] =
        dptr_dft_matrix_batch_RE_[i] + (16 * 16 * 16 * max_amount_of_warps[i]);

    ComputeDFTMatrix<<<max_amount_of_warps[i]*16, 16*16, 0, streams[i]>>>(
        dptr_dft_matrix_batch_RE_[i], dptr_dft_matrix_batch_IM_[i]);
  }

  std::vector<int> amount_of_transpose_blocks;
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    amount_of_transpose_blocks.push_back(
        ceil(static_cast<float>(fft_plans[i].fft_length_) /
             static_cast<float>(fft_plans[i].transposer_blocksize_)));
  }
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    TransposeKernel<<<amount_of_transpose_blocks[i],
                      fft_plans[i].transposer_blocksize_, 0, streams[i]>>>(
        dptr_input_RE[i], dptr_input_IM[i], dptr_results_RE[i],
        dptr_results_IM[i], fft_plans[i].fft_length_,
        fft_plans[i].amount_of_r16_steps_, fft_plans[i].amount_of_r2_steps_);
  }

  std::vector<int> amount_of_dft_blocks;
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    amount_of_dft_blocks.push_back(
        fft_plans[i].fft_length_ /
        (16 * 16 * 16 * fft_plans[i].dft_warps_per_block_));
  }

  //Launch baselayer DFT step kernel
  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    DFTKernel<<<amount_of_dft_blocks[i], 32 * fft_plans[i].dft_warps_per_block_,
               0, streams[i]>>>(
        dptr_input_RE[i], dptr_input_IM[i], dptr_results_RE[i],
        dptr_results_IM[i], dptr_dft_matrix_batch_RE_[i],
        dptr_dft_matrix_batch_IM_[i]);
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
        dptr_current_input_RE[i] = dptr_results_RE[i];
        dptr_current_input_IM[i] = dptr_results_IM[i];
        dptr_current_results_RE[i] = dptr_input_RE[i];
        dptr_current_results_IM[i] = dptr_input_IM[i];
      } else {
        dptr_current_input_RE[i] = dptr_input_RE[i];
        dptr_current_input_IM[i] = dptr_input_IM[i];
        dptr_current_results_RE[i] = dptr_results_RE[i];
        dptr_current_results_IM[i] = dptr_results_IM[i];
      }

      int amount_of_r16_blocks =
          fft_plans[i].fft_length_ / fft_plans[i].r2_blocksize_;

      shared_mem_in_bytes.push_back(fft_plans[i].r16_warps_per_block_ * 16 *
                                    16 * 16 * 2 * sizeof(__half));
      cudaFuncSetAttribute(Radix16Kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shared_mem_in_bytes[i]);

      Radix16Kernel<<<amount_of_r16_blocks,
                     32 * fft_plans[i].r16_warps_per_block_,
                     shared_mem_in_bytes[i], streams[i]>>>(
          dptr_current_input_RE[i], dptr_current_input_IM[i],
          dptr_current_results_RE[i], dptr_current_results_IM[i],
          dptr_dft_matrix_batch_RE_[i], dptr_dft_matrix_batch_IM_[i],
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
        dptr_current_input_RE[i] = dptr_results_RE[i];
        dptr_current_input_IM[i] = dptr_results_IM[i];
        dptr_current_results_RE[i] = dptr_input_RE[i];
        dptr_current_results_IM[i] = dptr_input_IM[i];
      } else {
        dptr_current_input_RE[i] = dptr_input_RE[i];
        dptr_current_input_IM[i] = dptr_input_IM[i];
        dptr_current_results_RE[i] = dptr_results_RE[i];
        dptr_current_results_IM[i] = dptr_results_IM[i];
      }

      int amount_of_r2_blocks =
          fft_plans[i].fft_length_ / fft_plans[i].r2_blocksize_;

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

  cudaDeviceSynchronize();

  for(int i=0; i<static_cast<int>(fft_plans.size()); i++){
    //Cpy results back to host
    if (cudaMemcpy(data[i], dptr_current_results_RE[i],
                    2 * fft_plans[i].fft_length_ * sizeof(__half),
                    cudaMemcpyDeviceToHost)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    //Free used device memory
    if (cudaFree(dptr_input_RE[i]) != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
    if (cudaFree(dptr_dft_matrix_batch_RE_[i]) != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }
  }

  return std::nullopt;
}
