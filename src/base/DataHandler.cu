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
//It is recommended to call PeakAtLastError() method after calling the
//constructor to check if the construction was successfull.
class DataHandler{
public:
  DataHandler(int fft_length) : fft_length_(fft_length) {
    if (cudaMalloc((void**)(&dptr_data_), 6 * sizeof(__half) * fft_length_)
        != cudaSuccess){
       std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    }
    dptr_input_RE_ = dptr_data_;
    dptr_input_IM_ = dptr_input_RE_ + fft_length_;
    dptr_results_RE_ = dptr_input_IM_ + fft_length_;
    dptr_results_IM_ = dptr_results_RE_ + fft_length_;
    dptr_dft_matrix_RE_ = dptr_results_IM_ + fft_length_;
    dptr_dft_matrix_IM_ = dptr_dft_matrix_RE_ + fft_length_;

    //Here we precompute the dft matrix batches needed for the DFTKernel() and
    //Radix16Kernel(). Currently there is one matrix precomputed for each warp.
    //The other options are to only precompute one (lower memory usage but read
    //conflicts for each warp) and to compute the dft matrix each time during the
    //kernels. (TODO: find out whats "best")
    ComputeDFTMatrix<<<fft_length / 256, 16*16>>>(dptr_dft_matrix_RE_,
                                                  dptr_dft_matrix_IM_);
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

  ~DataHandler(){
    cudaFree(dptr_data_);
  }

  int fft_length_;
  __half* dptr_data_;
  __half* dptr_input_RE_;
  __half* dptr_input_IM_;
  __half* dptr_results_RE_;
  __half* dptr_results_IM_;
  __half* dptr_dft_matrix_RE_;
  __half* dptr_dft_matrix_IM_;
};

//Similar to the DataHandler class but is used for the async fft compution and
//thus holds the data of the entire batch of ffts to be computed.
class DataBatchHandler{
public:
  DataBatchHandler(int fft_length, int amount_of_ffts) :
      fft_length_(fft_length), amount_of_ffts_(amount_of_ffts) {
    if (cudaMalloc((void**)(&dptr_data_),
                   amount_of_ffts_ * 6 * sizeof(__half) * fft_length_)
        != cudaSuccess){
       std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    }

    for(int i=0; i<amount_of_ffts_; i++){
      dptr_input_RE_.resize(amount_of_ffts_, nullptr);
      dptr_input_IM_.resize(amount_of_ffts_, nullptr);
      dptr_results_RE_.resize(amount_of_ffts_, nullptr);
      dptr_results_IM_.resize(amount_of_ffts_, nullptr);
      dptr_dft_matrix_RE_.resize(amount_of_ffts_, nullptr);
      dptr_dft_matrix_IM_.resize(amount_of_ffts_, nullptr);
    }
    for(int i=0; i<amount_of_ffts_; i++){
      dptr_input_RE_[i] = dptr_data_ + (2 * i * fft_length_);
      dptr_input_IM_[i] = dptr_input_RE_[i] + fft_length_;
    }
    for(int i=0; i<amount_of_ffts_; i++){
      dptr_results_RE_[i] = dptr_input_IM_[amount_of_ffts_ - 1] +
                            fft_length_ +
                            (2 * i * fft_length_);
      dptr_results_IM_[i] = dptr_results_RE_[i] + fft_length_;
    }
    for(int i=0; i<amount_of_ffts_; i++){
      dptr_dft_matrix_RE_[i] = dptr_results_IM_[amount_of_ffts_ - 1] +
                               fft_length_ +
                               (2 * i * fft_length_);
      dptr_dft_matrix_IM_[i] = dptr_dft_matrix_RE_[i] + fft_length_;
    }

    //Here we precompute the dft matrix batches needed for the DFTKernel() and
    //Radix16Kernel(). Currently there is one matrix precomputed for each warp.
    //The other options are to only precompute one (lower memory usage but read
    //conflicts for each warp) and to compute the dft matrix each time during the
    //kernels. (TODO: find out whats "best")
    //Create a stream for each fft
    std::vector<cudaStream_t> streams;
    streams.resize(amount_of_ffts_);
    for(int i=0; i<amount_of_ffts_; i++){
      cudaStreamCreate(&(streams[i]));
    }
    for(int i=0; i<amount_of_ffts_; i++){
      ComputeDFTMatrix<<<fft_length / 256, 16*16, 0, streams[i]>>>(
          dptr_dft_matrix_RE_[i], dptr_dft_matrix_IM_[i]);
    }
  }

  std::optional<std::string> PeakAtLastError() {
    if (cudaPeekAtLastError() != cudaSuccess){
      return cudaGetErrorString(cudaPeekAtLastError());
    }
    return std::nullopt;
  }

  std::optional<std::string> CopyDataHostToDevice(__half* data) {
    if (cudaMemcpy(dptr_input_RE_[0], data,
                   amount_of_ffts_ * 2 * fft_length_ * sizeof(__half),
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
    if (((amount_of_r16_steps + amount_of_r2_steps) % 2) == 1) {
      results = dptr_results_RE_[0];
    } else {
      results = dptr_input_RE_[0];
    }
    if (cudaMemcpy(data, results,
                   amount_of_ffts_ * 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyDeviceToHost)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    return std::nullopt;
  }

  ~DataBatchHandler(){
    cudaFree((void*)dptr_data_);
  }
  int fft_length_;
  int amount_of_ffts_;
  __half* dptr_data_;
  std::vector<__half*> dptr_input_RE_;
  std::vector<__half*> dptr_input_IM_;
  std::vector<__half*> dptr_results_RE_;
  std::vector<__half*> dptr_results_IM_;
  std::vector<__half*> dptr_dft_matrix_RE_;
  std::vector<__half*> dptr_dft_matrix_IM_;
};

//Multi GPU equivalent to the DataHandler class
//Supports only same length on all GPUs. Diff
class DataHandlerMultiGPU{
public:
  DataHandlerMultiGPU(int fft_length, std::vector<int> device_ids) :
      fft_length_(fft_length){
    device_ids_ = device_ids;
    dptr_data_.resize(device_ids_.size(), nullptr);
    dptr_input_RE_.resize(device_ids_.size(), nullptr);
    dptr_input_IM_.resize(device_ids_.size(), nullptr);
    dptr_results_RE_.resize(device_ids_.size(), nullptr);
    dptr_results_IM_.resize(device_ids_.size(), nullptr);
    dptr_dft_matrix_RE_.resize(device_ids_.size(), nullptr);
    dptr_dft_matrix_IM_.resize(device_ids_.size(), nullptr);

    for(int i=0; i<static_cast<int>(device_ids_.size()); i++){
      cudaSetDevice(device_ids_[i]);

      if (cudaMalloc((void**)(&dptr_data_[i]), 6 * sizeof(__half) * fft_length_)
          != cudaSuccess){
         std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
      }
      dptr_input_RE_[i] = dptr_data_[i];
      dptr_input_IM_[i] = dptr_input_RE_[i] + fft_length_;
      dptr_results_RE_[i] = dptr_input_IM_[i] + fft_length_;
      dptr_results_IM_[i] = dptr_results_RE_[i] + fft_length_;
      dptr_dft_matrix_RE_[i] = dptr_results_IM_[i] + fft_length_;
      dptr_dft_matrix_IM_[i] = dptr_dft_matrix_RE_[i] + fft_length_;

      //Here we precompute the dft matrix batches needed for the DFTKernel() and
      //Radix16Kernel(). Currently there is one matrix precomputed for each warp.
      //The other options are to only precompute one (lower memory usage but read
      //conflicts for each warp) and to compute the dft matrix each time during the
      //kernels. (TODO: find out whats "best")
      ComputeDFTMatrix<<<fft_length / 256, 16*16>>>(dptr_dft_matrix_RE_[i],
                                                    dptr_dft_matrix_IM_[i]);
    }
  }

  std::optional<std::string> PeakAtLastError() {
    for(int i=0; i<static_cast<int>(device_ids_.size()); i++){
      cudaSetDevice(device_ids_[i]);
      if (cudaPeekAtLastError() != cudaSuccess){
        return cudaGetErrorString(cudaPeekAtLastError());
      }
    }

    return std::nullopt;
  }

  //Copy data to device with id device_ids_[device_id]
  std::optional<std::string> CopyDataHostToDevice(__half* data, int device_id) {
    cudaSetDevice(device_ids_[device_id]);
    if (cudaMemcpy(dptr_input_RE_[device_id], data,
                   2 * fft_length_ * sizeof(__half), cudaMemcpyHostToDevice)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    return std::nullopt;
  }

  std::optional<std::string> CopyResultsDeviceToHost(__half* data,
                                                     int amount_of_r16_steps,
                                                     int amount_of_r2_steps,
                                                     int device_id) {
    cudaSetDevice(device_ids_[device_id]);

    __half* results;
    if (((amount_of_r16_steps + amount_of_r2_steps) % 2) != 0) {
      results = dptr_results_RE_[device_id];
    } else {
      results = dptr_input_RE_[device_id];
    }

    if (cudaMemcpy(data, results, 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyDeviceToHost)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    return std::nullopt;
  }

  ~DataHandlerMultiGPU(){
    for(int i=0; i<static_cast<int>(device_ids_.size()); i++){
      cudaSetDevice(device_ids_[i]);
      cudaFree(dptr_data_[i]);
    }
  }

  int fft_length_;
  std::vector<int> device_ids_;
  std::vector<__half*> dptr_data_;
  std::vector<__half*> dptr_input_RE_;
  std::vector<__half*> dptr_input_IM_;
  std::vector<__half*> dptr_results_RE_;
  std::vector<__half*> dptr_results_IM_;
  std::vector<__half*> dptr_dft_matrix_RE_;
  std::vector<__half*> dptr_dft_matrix_IM_;
};

class DataBatchHandlerMultiGPU{
public:
  DataBatchHandlerMultiGPU(int fft_length, int amount_of_ffts,
                           std::vector<int> device_ids) :
      fft_length_(fft_length), amount_of_ffts_(amount_of_ffts) {
    device_ids_ = device_ids;
    dptr_data_.resize(device_ids_.size());
    dptr_input_RE_.resize(device_ids_.size());
    dptr_input_IM_.resize(device_ids_.size());
    dptr_results_RE_.resize(device_ids_.size());
    dptr_results_IM_.resize(device_ids_.size());
    dptr_dft_matrix_RE_.resize(device_ids_.size());
    dptr_dft_matrix_IM_.resize(device_ids_.size());
    for(int i=0; i<static_cast<int>(device_ids_.size()); i++){
      dptr_input_RE_[i].resize(amount_of_ffts_, nullptr);
      dptr_input_IM_[i].resize(amount_of_ffts_, nullptr);
      dptr_results_RE_[i].resize(amount_of_ffts_, nullptr);
      dptr_results_IM_[i].resize(amount_of_ffts_, nullptr);
      dptr_dft_matrix_RE_[i].resize(amount_of_ffts_, nullptr);
      dptr_dft_matrix_IM_[i].resize(amount_of_ffts_, nullptr);
    }

    for(int j=0; j<static_cast<int>(device_ids_.size()); j++){
      cudaSetDevice(device_ids_[j]);

      if (cudaMalloc((void**)(&dptr_data_[j]),
                     amount_of_ffts_ * 6 * sizeof(__half) * fft_length_)
          != cudaSuccess){
         std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
      }

      for(int i=0; i<amount_of_ffts_; i++){
        dptr_input_RE_[j][i] = dptr_data_[j] + (2 * i * fft_length_);
        dptr_input_IM_[j][i] = dptr_input_RE_[j][i] + fft_length_;
      }
      for(int i=0; i<amount_of_ffts_; i++){
        dptr_results_RE_[j][i] = dptr_input_IM_[j][amount_of_ffts_ - 1] +
                              fft_length_ +
                              (2 * i * fft_length_);
        dptr_results_IM_[j][i] = dptr_results_RE_[j][i] + fft_length_;
      }
      for(int i=0; i<amount_of_ffts_; i++){
        dptr_dft_matrix_RE_[j][i] = dptr_results_IM_[j][amount_of_ffts_ - 1] +
                                 fft_length_ +
                                 (2 * i * fft_length_);
        dptr_dft_matrix_IM_[j][i] = dptr_dft_matrix_RE_[j][i] + fft_length_;
      }

      //Here we precompute the dft matrix batches needed for the DFTKernel() and
      //Radix16Kernel(). Currently there is one matrix precomputed for each warp.
      //The other options are to only precompute one (lower memory usage but read
      //conflicts for each warp) and to compute the dft matrix each time during the
      //kernels. (TODO: find out whats "best")
      //Create a stream for each fft
      std::vector<cudaStream_t> streams;
      streams.resize(amount_of_ffts_);
      for(int i=0; i<amount_of_ffts_; i++){
        cudaStreamCreate(&(streams[i]));
      }
      for(int i=0; i<amount_of_ffts_; i++){
        ComputeDFTMatrix<<<fft_length / 256, 16*16, 0, streams[i]>>>(
            dptr_dft_matrix_RE_[j][i], dptr_dft_matrix_IM_[j][i]);
      }
    }

  }

  std::optional<std::string> PeakAtLastError() {
    for(int i=0; i<static_cast<int>(device_ids_.size()); i++){
      cudaSetDevice(device_ids_[i]);
      if (cudaPeekAtLastError() != cudaSuccess){
        return cudaGetErrorString(cudaPeekAtLastError());
      }
    }

    return std::nullopt;
  }

  //Copy data to device with id device_ids_[device_id]
  std::optional<std::string> CopyDataHostToDevice(__half* data, int device_id){
    cudaSetDevice(device_ids_[device_id]);
    if (cudaMemcpy(dptr_input_RE_[device_id][0], data,
                   amount_of_ffts_ * 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyHostToDevice)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    return std::nullopt;
  }

  std::optional<std::string> CopyResultsDeviceToHost(__half* data,
                                                     int amount_of_r16_steps,
                                                     int amount_of_r2_steps,
                                                     int device_id) {
    cudaSetDevice(device_ids_[device_id]);

    __half* results;
    if (((amount_of_r16_steps + amount_of_r2_steps) % 2) == 1) {
      results = dptr_results_RE_[device_id][0];
    } else {
      results = dptr_input_RE_[device_id][0];
    }

    if (cudaMemcpy(data, results,
                   amount_of_ffts_ * 2 * fft_length_ * sizeof(__half),
                   cudaMemcpyDeviceToHost)
         != cudaSuccess) {
       return cudaGetErrorString(cudaPeekAtLastError());
    }

    return std::nullopt;
  }

  ~DataBatchHandlerMultiGPU(){
    for(int i=0; i<static_cast<int>(device_ids_.size()); i++){
      cudaSetDevice(device_ids_[i]);
      cudaFree((void*)dptr_data_[i]);
    }
  }

  int fft_length_;
  int amount_of_ffts_;
  std::vector<int> device_ids_;
  std::vector<__half*> dptr_data_;
  std::vector<std::vector<__half*>> dptr_input_RE_;
  std::vector<std::vector<__half*>> dptr_input_IM_;
  std::vector<std::vector<__half*>> dptr_results_RE_;
  std::vector<std::vector<__half*>> dptr_results_IM_;
  std::vector<std::vector<__half*>> dptr_dft_matrix_RE_;
  std::vector<std::vector<__half*>> dptr_dft_matrix_IM_;
};
