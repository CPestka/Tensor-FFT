#pragma once

//Used to test functonality
#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../../base/Plan.cpp"
#include "../../base/ComputeFFT.cu"
#include "../TestingDataCreation.cu"
#include "../FileWriter.cu"

bool full_test_16(){
  int fft_length = 16*16;

  //Prepare input data on cpu
  __half* data;
  std::vector<float> weights;
  weights.push_back(1.0);
  data = CreateSineSuperpostion(fft_length, weights).get();

  //Get plan
  Plan my_plan;
  if (CreatePlan(fft_length)) {
    my_plan = CreatePlan(fft_length).value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  //Construct a DataHandler for data on GPU
  DataHandler my_handler(fft_length);
  if (my_handler.PeakAtLastError() != cudaSuccess) {
    std::cout << "Memory allocation on device failed." << std::endl;
    return false;
  }

  std::string error_mess;

  //Copy data to gpu
  error_mess = my_handler.CopyDataHostToDevice(data).value();
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  //Compute FFT
  error_mess = ComputeFFT(my_plan, my_handler).value();
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  //Copy results back to cpu
  error_mess = my_handler.CopyResultsDevicetoHost(data).value();
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  WriteResultsToFile("test_fft_16.dat", fft_length, data);

  return true;
}
