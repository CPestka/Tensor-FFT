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
#include "../Timer.h"

bool full_test(int fft_length, std::string file_name_prefix){
  IntervallTimer my_timer();

  //Prepare input data on cpu
  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data = CreateSineSuperpostion(fft_length, weights);

  //Get plan
  Plan my_plan;
  if (CreatePlan(fft_length)) {
    my_plan = CreatePlan(fft_length).value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  WriteResultsREToFile(file_name_prefix + "_input.dat", fft_length, data.get());

  std::string error_mess;

  //Construct a DataHandler for data on GPU
  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError().value_or("");
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  //Copy data to gpu
  error_mess = my_handler.CopyDataHostToDevice(data.get()).value_or("");
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  //Compute FFT
  error_mess = ComputeFFT(my_plan, my_handler).value_or("");
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  //Copy results back to cpu
  error_mess = my_handler.CopyResultsDeviceToHost(
      data.get(), my_plan.amount_of_r16_steps_,
      my_plan.amount_of_r2_steps_).value_or("");
  if (error_mess != "") {
    std::cout << error_mess << std::endl;
    return false;
  }

  cudaDeviceSynchronize();

  WriteResultsToFile(file_name_prefix + "_results.dat", fft_length, data.get());

  my_timer.getTimeInMilliseconds();

  return true;
}
