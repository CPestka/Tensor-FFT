//Shows via a simple example how to compute FFTs via the provided functions
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>
#include <cassert>
#include <cinttypes>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.h"
#include "FileWriter.h"
#include "Timer.h"
#include "../base/ComputeFFT.h"
#include "../base/Plan.h"

int main(){
  constexpr int fft_length = 16*16*16*16*16 * 16*2;

  //Creation of example data
  //Substitute your own real data here. Data is accepted as __half array with
  //fft_length*2 amount of elements, with the RE elements making up the first
  //and the IM elements making up the second half of the array.
  std::vector<float> weights_RE { 1.0, 0.7, 0.5, 0.2, 0.3, 0.7, 0.8 };
  std::vector<float> weights_IM { 1.0, 0.3, 0.2, 0.4, 0.9, 0.1, 0.6 };
  // std::vector<float> weights_RE { 1.0 };
  // std::vector<float> weights_IM { 0.0 };
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionHGPU(fft_length, weights_RE, weights_IM, 7);

  //Write results to file
  //WriteResultsToFile("example_in.dat", fft_length, data.get());

  std::optional<std::string> error_mess;

  //The plan holds parameters needed for the execution of the kernels which are
  //mostly derived from the fft length.
  std::optional<Plan<int>> possible_plan = CreatePlan(fft_length, Mode_4096, 16, 16, 512);
  Plan<int> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  //Check if parameters of plan work given limitations on used device.
  int device_id;
  cudaGetDevice(&device_id);
  assert((PlanWorksOnDevice(my_plan, device_id)));

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
  error_mess = ComputeFFT(my_plan, my_handler,
                          GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Copy results back
  error_mess =
    my_handler.CopyResultsDeviceToHost(data.get(),
                                       my_plan.results_in_results_);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  //Make sure the results have finished cpying
  cudaDeviceSynchronize();

  //Write results to file
  //WriteResultsToFile("example_results.dat", fft_length, data.get());

  return true;
}
