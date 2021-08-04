//Shows via a simple example how to compute a batch of FFTs asynchronously
//Similar to ExampleSingleFFT.cu see there for more details
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
  int fft_length = 16*16*16;
  int batch_size = 20;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionBatch(fft_length, batch_size, weights);

  std::optional<std::string> error_mess;

  Plan my_plan;
  if (CreatePlan(fft_length)) {
    my_plan = CreatePlan(fft_length).value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  //Instead of the DataHandler class
  DataBatchHandler my_handler(fft_length, batch_size);
  error_mess = my_handler.PeakAtLastError();
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  error_mess = my_handler.CopyDataHostToDevice(data.get());
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Instead of the ComputeFFT() function
  error_mess = ComputeFFTs(my_plan, my_handler);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  /*
  error_mess =
    my_handler.CopyResultsDeviceToHost(data.get(), my_plan.amount_of_r16_steps_,
                                       my_plan.amount_of_r2_steps_);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  cudaDeviceSynchronize();

  //Write results to file
  WriteResultBatchToFile( , fft_length, data.get());
  */

  return true;
}
