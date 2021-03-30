//Used to test functonality
#include <iostream>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../base/GraphHandler.cu"
#include "TestingDataCreation.cu"
#include "../base/Timer.h"

int main(){
  //Data parameter
  int fft_length = 16*16*16;
  int amount_of_oscialtions = 1;
  //Kernel / performance parameter
  int amount_host_to_device_memcopies = 1;
  int dft_max_warps = 16;
  int dft_max_blocks = 256;
  int radix16_max_warps = 16;
  int radix16_max_blocks = 256;
  int radix2_max_blocksize = 512;
  int radix2_max_blocks = 256;
  int transpose_blocksize = 512;
  int transpose_amount_of_blocks_per_kernel = 256;
  int radix2_loop_length = 64;

  //Create simple example data
  std::unique_ptr<__half2[]> data = CreateRealCosineData(amount_of_oscialtions,
                                                       fft_length);
  //Allocate output array
  std::unique_ptr<__half[]> results =
      std::make_unique<__half[]>(2 * fft_length);
  
  //Create graph via constructing a GraphHandler
  IntervallTimer my_intervall_timer;
  GraphHandler my_graph_handler(
      fft_length, data.get(), results.get(),
      amount_host_to_device_memcopies, dft_max_warps,
      dft_max_blocks, radix16_max_warps, radix16_max_blocks,
      radix2_max_blocksize, radix2_max_blocks, radix2_loop_length,
      transpose_blocksize, transpose_amount_of_blocks_per_kernel);
  int64_t setup_time_ms = my_intervall_timer.getTimeInMilliseconds();
  std::cout << "Setup took " << setup_time_ms << " ms" << std::endl;

  //Executing graph i.e. performing the fft
  my_graph_handler.ExecuteGraph();
  int64_t fft_time_ms = my_intervall_timer.getTimeInMilliseconds() -
                        setup_time_ms;
  std::cout << "FFT took " << fft_time_ms << " ms" << std::endl;

  //Returning results to host
  my_graph_handler.CopyResultsDevicetoHost();
  int64_t returning_results_time_ms = my_intervall_timer.getTimeInMilliseconds()
                                      - setup_time_ms - fft_time_ms;
  std::cout << "Copying results back took " << returning_results_time_ms
            << " ms" << std::endl;
  return true;
}
