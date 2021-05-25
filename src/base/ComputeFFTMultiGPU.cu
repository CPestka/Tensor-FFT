//Multi GPU version of the function ComputeFFTs()
#pragma once
#include <vector>
#include <optional>
#include <string>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ComputeFFT.cu"
#include "../testing/FileWriter.cu"

void SingleGPUWork(int device_id, std::vector<Plan> fft_plans,
                   std::vector<__half*> data, std::vector<std::string>* error){
  cudaSetDevice(device_id);

  std::optional<std::string> single_GPU_error = ComputeFFTs(fft_plans, data);

  if (single_GPU_error.has_value) {
    WriteLogToFile("Device" + to_string(device_id) + "Error.log",
                   single_GPU_error.value);
  }
}

//Simple implementation for multiple GPUs of ComputeFFTs(). device_list[i]
//contains the id of the GPU to be used for the call of
//ComputeFFTs(fft_plans[i], data[i]). The ids HAVE to be unique and correspond
//to an existing device id of the used system.
//Errors are writen to log files.
void ComputeFFTsMultiGPU(std::vector<int> device_list,
                         std::vector<std::vector<Plan>> fft_plans,
                         std::vector<std::vector<__half*>> data){
  std::vector<thread> worker;
  for(int i=0; i<static_cast<int>(device_list.size()); i++){
    worker.push_back(std::thread(&SingleGPUWork, device_list[i], fft_plans[i],
                                 data[i]));
  }

  for(int i=0; i<static_cast<int>(device_list.size()); i++){
    worker[i].join();
  }
}
