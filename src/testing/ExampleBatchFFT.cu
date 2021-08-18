//Shows via a simple example how to compute a batch of FFTs asynchronously
//Similar to ExampleSingleFFT.cu see there for more details
#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <memory>
#include <cassert>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.h"
#include "FileWriter.h"
#include "Timer.h"
#include "../base/ComputeFFT.h"
#include "../base/Plan.h"


int main(){
  constexpr int fft_length = 16*16*16;
  constexpr int batch_size = 20;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data =
      CreateSineSuperpostionBatch(fft_length, batch_size, weights);

  std::optional<std::string> error_mess;

  std::optional<Plan<int>> possible_plan = CreatePlan(fft_length);
  Plan<int> my_plan;
  if (possible_plan) {
    my_plan = possible_plan.value();
  } else {
    std::cout << "Plan creation failed" << std::endl;
    return false;
  }

  int device_id;
  cudaGetDevice(&device_id);
  assert((PlanWorksOnDevice(my_plan, device_id)));

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

  //Uses different overload
  error_mess = ComputeFFT(my_plan, my_handler,
                          GetMaxNoOptInSharedMem(device_id));
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  error_mess =
    my_handler.CopyResultsDeviceToHost(data.get(),
                                       fft_plan.results_in_results_);
  if (error_mess) {
    std::cout << error_mess.value() << std::endl;
    return false;
  }

  cudaDeviceSynchronize();

  //Write results to file
  WriteResultBatchToFile( , fft_length, data.get());

  return true;
}
