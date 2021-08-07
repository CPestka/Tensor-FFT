//Shows via a simple example how to compute FFTs via the provided functions
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.cu"
#include "FileWriter.cu"
#include "Timer.h"
#include "../base/ComputeFFT.cu"
#include "../base/Plan.cpp"


int main(){
  int fft_length = 16*16*16*16*16;

  //Creation of example data
  //Substitute your own real data here. Data is accepted as __half array with
  //fft_length*2 amount of elements, with the RE elements making up the first
  //and the IM elements making up the second half of the array.
  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostion(fft_length, weights);

  std::optional<std::string> error_mess;

  //The plan holds parameters needed for the execution of the kernels which are
  //mostly derived from the fft length.
  Plan my_plan;
  if (CreatePlan(fft_length)) {
    my_plan = CreatePlan(fft_length).value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  //The DataHandler class allocates and holds the ptrs to the data on the device
  //Instantiation and destruction handle the allocation and freeing of the
  //needed memory on the device.
  DataHandler my_handler(fft_length);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Copy input data to device
  error_mess = my_handler.CopyDataHostToDevice(data.get());
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Compute the FFT on the device
  error_mess = ComputeFFT(my_plan, my_handler);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  cudaDeviceSynchronize();

  /*
  //Copy results back
  error_mess =
    my_handler.CopyResultsDeviceToHost(data.get(), my_plan.amount_of_r16_steps_,
                                       my_plan.amount_of_r2_steps_);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  //Write results to file
  WriteResultsToFile("example_results.dat", fft_length, data.get());
  */

  return true;
}
