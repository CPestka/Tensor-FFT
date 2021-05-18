//Contains mostly the GraphHandler class which is used to create a graph of
//memory copies and kernel launches needed to perform the FFT of a specified
//input and execute it.
#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "Timer.h"
#include "Transposer.cu"
#include "TensorDFT16.cu"
#include "TensorRadix16.cu"
#include "Radix2.cu"

bool IsPowerOf2(int x) {
  if (x==0){
    return false;
  }
  return ((x & (x - 1)) == 0);
}

//Requires x to be power of 2
int ExactLog2(int x) {
  if (x == 1) {
    return 0;
  }

  int tmp = x;
  int i = 1;

  while (true) {
    if (((tmp/2) % 2) == 0) {
      i++;
      tmp = tmp / 2;
    } else {
      return i;
    }
  }
}

//Contains the needed information for the creation of the transpose kernels
struct TransposeLaunchConfig {
  int blocksize_;
  int amount_of_blocks_per_kernel_;
  int amount_of_kernels_;
  std::vector<int> kernel_ids_;
};

//Contains the needed information for the creation of the kernels of the base
//layer DFT step of the FFT
struct DFTLaunchConfig {
  int amount_of_ffts_;
  int amount_of_warps_per_block_;
  int blocksize_;
  int amount_of_blocks_per_kernel_;
  int amount_of_matrices_per_warp_;
  int amount_of_kernels_;
  std::vector<int> kernel_ids_;
};

//Contains the needed information for the creation of the kernels of the radix16
//steps of the FFT
struct Radix16LaunchConfig {
  int current_radix16_step_;
  int size_of_ffts_;
  int amount_of_ffts_;
  int amount_of_warps_per_block_;
  int blocksize_;
  int amount_of_blocks_per_kernel_;
  int amount_of_matrices_per_warp_;
  int amount_of_kernels_;
  std::vector<int> kernel_ids_;
};

//Contains the needed information for the creation of the kernels of the radix2
//steps of the FFT
struct Radix2LaunchConfig {
  int current_radix2_step_;
  int amount_of_ffts_;
  int size_of_ffts_;
  int blocksize_;
  int amount_of_blocks_per_kernel_;
  int amount_of_kernels_;
  std::vector<int> kernel_ids_;
  std::vector<int> kernel_memory_offfset_;
  int amount_of_kernels_per_fft_;
};

//This class is used to perform the fft of a given input.
//How to use: 1. Call Constructor (internaly creates graph of kernels)
//            2. Call ExecuteGraph() which executes the graph if the creation
//               of the graph was successfull (can also be checked via
//               IsGraphExecutable())
//            3. Call CopyResultsDevicetoHost() to transfer results back to host
//The main parameter for the constructor are the fft lenght and a ptr to a
//__half2 array which holds the complex fp16 input data.
//The results are written into the __half[2*fft_length_] array results with all
//RE in the first half of the array and all IM in the second part.
//This FFT implementation utilizes the radix16 and radix2 variants of the
//Cooley-Tukey algorithm and implements them on the GPU, with the radix16 part
//of the calculation and the base layer DFT step being accelerated by tensor
//cores. Due to the used algorithms the input size is restricted to powers of 2.
//Expressing the input size N=2**M=2**K * 16**L, where K is the minimal value
//for a given N,comparative performance is best for large values of L and small
//fractions K/L, due to the fact that the radix2 part of the algorithm is not
//accelerated by tensor cores.
//The other parameters of the constructor are performance parameter. For more
//detail on them and some restrictions on them rising from the fft lenght refere
//to the functions Create...LaunchConfig().
//During destruction GraphHandler wont free memory of data and results.
class GraphHandler {
public:
  GraphHandler(int fft_length, __half2* data, __half* results,
               int amount_host_to_device_memcopies, int dft_max_warps,
               int dft_max_blocks, int radix16_max_warps,
               int radix16_max_blocks, int radix2_max_blocksize,
               int radix2_max_blocks, int transpose_blocksize,
               int transpose_amount_of_blocks_per_kernel)
      : fft_length_(fft_length), graph_is_valid_(false), hptr_data_(data),
        hptr_results_RE_(results), hptr_results_IM_(results+fft_length) {

    //Consequtively parse input length, set up launch configs and create graph
    //if the previous step was successfull
    bool tmp = ParseInputLength();
    if (tmp) {
      tmp = CreateTransposeLaunchConfig(amount_host_to_device_memcopies,
                                        transpose_blocksize,
                                        transpose_amount_of_blocks_per_kernel);
    }
    if (tmp) {
      tmp = CreateDFTLaunchConfig(dft_max_warps, dft_max_blocks);
    }
    if (tmp) {
      tmp = CreateRadix16LaunchConfig(radix16_max_warps, radix16_max_blocks);
    }
    if (tmp) {
      tmp = CreateRadix2LaunchConfig(radix2_max_blocksize, radix2_max_blocks);
    }
    if (tmp) {
      CreateGraph();
    } else {
      std::cout << "Graph could not be created" << std::endl;
    }
  }

  ~GraphHandler(){
    //Destroy used graphs and streams
    if (cudaGraphExecDestroy(executable_fft_graph_) != cudaSuccess) {
      std::cout << "Error! Destroying executable_fft_graph_ failed."
                << std::endl;
    }
    if (cudaGraphDestroy(fft_graph_) != cudaSuccess) {
      std::cout << "Error! Destroying fft_graph_ failed."
                << std::endl;
    }
    if (cudaGraphDestroy(memcpy_transpose_child_graph_) != cudaSuccess) {
      std::cout << "Error! Destroying fft_graph_ failed."
                << std::endl;
    }
    if (cudaStreamDestroy(fft_stream_) != cudaSuccess) {
      std::cout << "Error! Destroying fft_stream_ failed."
                << std::endl;
    }

    //Free used device memory
    cudaFree(dptr_data_);
    cudaFree(dptr_results_);
  }

  bool IsGraphExecutable(){
    return graph_is_valid_;
  }

  bool ExecuteGraph(){
    if (!graph_is_valid_) {
      std::cout << "Error! Graph can not be executed since it isnt vaild."
                << std::endl;
      return false;
    }

    if (cudaStreamCreateWithFlags(&fft_stream_, cudaStreamNonBlocking)
        != cudaSuccess) {
      std::cout << "Error! Creating stream for fft failed."
                << std::endl;
      return false;
    }

    if (cudaGraphLaunch(executable_fft_graph_, fft_stream_) != cudaSuccess) {
      std::cout << "Error! Launching executable graph failed."
                << std::endl;
      return false;
    }

    if (cudaStreamSynchronize(fft_stream_) != cudaSuccess) {
      std::cout << "Error! Synchronizing fft stream failed."
                << std::endl;
      return false;
    }

    std::cout << "Graph execution successfull!" << std::endl;

    return true;
  }

  //Copies finished fft back to host into the hptr_results_x_ arrays.
  //Check if graph executed succesfully for the fft in question before calling.
  bool CopyResultsDevicetoHost(){
    if (((amount_of_radix_16_steps_ + amount_of_radix_2_steps_) % 2) == 0) {
      if (cudaMemcpy(hptr_results_RE_, dptr_data_RE_,
                     fft_length_ * 2 * sizeof(__half), cudaMemcpyDeviceToHost)
          != cudaSuccess) {
        std::cout << "Error! Copying results to Host has failed."
                  << std::endl;
        return false;
      }
    } else {
      if (cudaMemcpy(hptr_results_RE_, dptr_results_RE_,
                     fft_length_ * 2 * sizeof(__half), cudaMemcpyDeviceToHost)
          != cudaSuccess) {
        std::cout << "Error! Copying results to Host has failed."
                  << std::endl;
        return false;
      }
    }

    std::cout << "Device to Host memcpy successfull!" << std::endl;

    return true;
  }

private:
  int fft_length_;
  //i.e. floor(log16(fft_length))-1
  int amount_of_radix_16_steps_;
  //i.e. log2(fft_length) - 4*floor(log16(fft_length))
  int amount_of_radix_2_steps_;
  //used to signal whether the graph creation was successfull
  bool graph_is_valid_;
  TransposeLaunchConfig transpose_conf_;
  DFTLaunchConfig dft_conf_;
  std::vector<Radix16LaunchConfig> radix16_conf_;
  std::vector<Radix2LaunchConfig> radix2_conf_;
  __half2* hptr_data_; //holds the input data on the CPU
  //results are returned as two __half arrays instead of one __half2 array
  __half* hptr_results_RE_;
  __half* hptr_results_IM_;
  //These two ptrs point to arrays that hold the data on the device during the
  //calculation. The data is writen back and forth between the two arrays during
  //the fft steps (i.e. the used algorithm is a out of place algorithm)
  __half2* dptr_data_;
  __half2* dptr_results_;
  //During its time on the device the data is unpacked and the RE and IM are
  //stored seperatly for performance reasons. dptr_x_RE_ + dptr_x_IM_ use the
  //same memory as dptr_x_
  __half* dptr_data_RE_;
  __half* dptr_data_IM_;
  __half* dptr_results_RE_;
  __half* dptr_results_IM_;
  __half* dptr_dft_matrix_batch_RE_;
  __half* dptr_dft_matrix_batch_IM_;
  __half hptr_dft_matrix_batch_RE_[16*16*16];
  __half hptr_dft_matrix_batch_IM_[16*16*16];
  cudaGraph_t fft_graph_; //graph that holds all work to be executed
  cudaStream_t fft_stream_; //stream in which fft_graph_ will be placed
  cudaGraphExec_t executable_fft_graph_;
  //Nodes and parameters of those needed for the fft_graph_
  //More details in CreateGraph()
  cudaGraph_t memcpy_transpose_child_graph_;
  cudaGraphNode_t memcpy_transpose_child_graph_node_;
  cudaGraphNode_t dft_matrix_batch_memcopy_;
  std::vector<cudaGraphNode_t> memcopies_;
  std::vector<cudaGraphNode_t> transpose_kernels_;
  std::vector<cudaKernelNodeParams> transpose_kernel_params_;
  std::vector<std::vector<void*>> transpose_kernel_args_;
  std::vector<cudaGraphNode_t> dft_kernels_;
  std::vector<cudaKernelNodeParams> dft_kernel_params_;
  std::vector<std::vector<void*>> dft_kernel_args_;
  std::vector<std::vector<cudaGraphNode_t>> radix16_kernels_;
  std::vector<std::vector<cudaKernelNodeParams>> radix16_kernel_params_;
  std::vector<std::vector<std::vector<void*>>> radix16_kernel_args_;
  std::vector<std::vector<cudaGraphNode_t>> radix2_kernels_;
  std::vector<std::vector<cudaKernelNodeParams>> radix2_kernel_params_;
  std::vector<std::vector<std::vector<void*>>> radix2_kernel_args_;

  //Determines the amount of radix16 and radix2 steps.
  //Since the base dft and radix 16 step utilize 16x16 matrix multiplications
  //performed by tensor cores, which due to a wwmma api limitation have to qued
  //in batches of 16, the input size of the data has to be devisable by 16*16*16
  //=4096 (thus also the input size has to be larger than that).
  bool ParseInputLength(){
    if (((fft_length_ % 4096) != 0) || !(IsPowerOf2(fft_length_))) {
      std::cout << "Input size is NOT a power of 2 or devisable by 4096! This "
                << "algorithm only supports inputsizes that are >= 4096"
                << " and powers of 2."
                << std::endl;
      return false;
    } else {
      int log2 = ExactLog2(fft_length_);
      amount_of_radix_16_steps_ = (log2 / 4) - 1;
      amount_of_radix_2_steps_ = log2 % 4;

      std::cout << "FFT lenght: " << fft_length_ << " = 2**" << log2 << " = 2**"
                << amount_of_radix_2_steps_ << " * 16**" << log2 / 4
                << std::endl
                << "Amount of DFT steps: 1"
                << std::endl
                << "Amount of radix 16 steps: " << amount_of_radix_16_steps_
                << std::endl
                << "Amount of radix 2 steps: " << amount_of_radix_2_steps_
                << std::endl;

      std::cout << "Input parsing successfull!" << std::endl;

      return true;
    }
  }

  //Determines launch configuration of transpose kernel
  bool CreateTransposeLaunchConfig(int amount_of_mem_copies, int blocksize,
                                   int amount_of_blocks_per_kernel){
    if ((fft_length_ % amount_of_mem_copies) != 0) {
      std::cout << "Error! The fft length has to be devisable by amount_of_mem_"
                << "copies_."
                << std::endl;
      return false;
    }

    transpose_conf_.blocksize_ = blocksize;
    transpose_conf_.amount_of_blocks_per_kernel_ = amount_of_blocks_per_kernel;
    transpose_conf_.amount_of_kernels_ = amount_of_mem_copies;
    for(int i=0; i<amount_of_mem_copies; i++){
      transpose_conf_.kernel_ids_.push_back(i);
    }

    std::cout << "Creating transpose config successfull!" << std::endl;

    return true;
  }

  //Determines the number kernels and their launch configuration based on the
  //fft length and the parameters dft_max_warps and dft_max_blocks.
  //The fft length determines how many warps in total have to be launched. The
  //current behavior is that depending on the size of that number the amount of
  //warps per block grows until it reaches dft_max_warps and analogously after
  //that the amount of blocks per kernel until it reaches dft_max_blocks. If
  //there are more blocks in total than dft_max_blocks multiple kernels will be
  //launched.
  //The parameter dft_max_warps can be increased to increase ocupancy
  //on one SM but this is limited by avaiable recources (registers,
  //tensor cores,...). Increasing dft_max_blocks will cause better ocupancy on
  //the total amount of SMs per GPU but decreasing it will cause the launch of
  //more kernels which will make it possible to execute further steps (i.e. the
  //radix steps) on data produced by kernels that have already finished.
  //More information about the performance implications of launch configurations
  //can be found in the nvidia proframming guide for cuda.
  bool CreateDFTLaunchConfig(int dft_max_warps, int dft_max_blocks){
    dft_conf_.amount_of_ffts_ = fft_length_ / 16;
    dft_conf_.amount_of_matrices_per_warp_ = 16; //required by current wwma api

    //Determines the amount of warps per block and thus also the blocksize
    if ((dft_conf_.amount_of_ffts_ / (16 * 16)) <= dft_max_warps) {
      dft_conf_.amount_of_warps_per_block_ =
          dft_conf_.amount_of_ffts_ / (16 * 16);
    } else {
      dft_conf_.amount_of_warps_per_block_ = dft_max_warps;
    }
    dft_conf_.blocksize_ = dft_conf_.amount_of_warps_per_block_ * 32;

    //Determines the amount of blocks per kernel
    if (dft_conf_.amount_of_warps_per_block_ < dft_max_warps){
      dft_conf_.amount_of_blocks_per_kernel_ = 1;
    } else {
      if (((dft_conf_.amount_of_ffts_ / (16 * 16)) % dft_max_warps) != 0) {
        std::cout << "Error! The total amount of warps has to be devisable by "
                  << "dft_max_warps (i.e. (fft_length/256)%dft_max_warps == 0)."
                  << std::endl;
        return false;
      } else {
        if (((dft_conf_.amount_of_ffts_ / (16 * 16)) / dft_max_warps) <=
            dft_max_blocks) {
          dft_conf_.amount_of_blocks_per_kernel_ =
              (dft_conf_.amount_of_ffts_ / (16 * 16)) / dft_max_warps;
        } else {
          dft_conf_.amount_of_blocks_per_kernel_ = dft_max_blocks;
        }
      }
    }

    //determines the amount of dft kernels to be launched
    if (dft_conf_.amount_of_blocks_per_kernel_ < dft_max_blocks) {
      dft_conf_.amount_of_kernels_ = 1;
    } else {
      if (((dft_conf_.amount_of_ffts_ / (16 * 16 * dft_max_warps)) %
          dft_max_blocks) != 0) {
        std::cout << "Error! The total amount of blocks has to be devisable by "
                  << "dft_max_blocks (i.e. "
                  << "(fft_length/(256*dft_max_warps))%dft_max_blocks == 0)."
                  << std::endl;
        return false;
      } else {
        dft_conf_.amount_of_kernels_ =
            (dft_conf_.amount_of_ffts_ / (16 * 16 * dft_max_warps)) %
            dft_max_blocks;
      }
    }
    for(int i=0; i<dft_conf_.amount_of_kernels_; i++){
      dft_conf_.kernel_ids_.push_back(i);
    }

    std::cout << "Creating DFT config successfull!" << std::endl;

    return true;
  }

  //Anlalogous to CreateDFTLaunchConfig but for the radix16 steps
  bool CreateRadix16LaunchConfig(int radix16_max_warps, int radix16_max_blocks){
    radix16_conf_.resize(amount_of_radix_16_steps_);
    for(int i=0; i<amount_of_radix_16_steps_; i++){
      radix16_conf_[i].amount_of_matrices_per_warp_ = 16;//required by wwma api
      radix16_conf_[i].current_radix16_step_ = i;
      radix16_conf_[i].size_of_ffts_ = std::pow(16,i+1);
      radix16_conf_[i].amount_of_ffts_ =
          (fft_length_ / radix16_conf_[i].size_of_ffts_);

      //Determine amount of warps per block
      if ((fft_length_ / (16 * 16 * 16)) <= radix16_max_warps) {
        radix16_conf_[i].amount_of_warps_per_block_ =
            fft_length_ / (16 * 16 * 16);
      } else {
        radix16_conf_[i].amount_of_warps_per_block_ = radix16_max_warps;
      }
      radix16_conf_[i].blocksize_ =
          radix16_conf_[i].amount_of_warps_per_block_ * 32;

      //Determine amount of blocks per kernel
      if (radix16_conf_[i].amount_of_warps_per_block_ < radix16_max_warps) {
        radix16_conf_[i].amount_of_blocks_per_kernel_ = 1;
      } else {
        if (((fft_length_ / (16 * 16 * 16)) % radix16_max_warps) != 0) {
          std::cout << "Error! The total amount of warps has to be devisable by"
                    << " radix16_max_warps "
                    << "(i.e. (fft_length/256)%radix16_max_warps == 0)."
                    << std::endl;
          return false;
        } else {
          if (((fft_length_ / (16 * 16 * 16)) / radix16_max_warps) <=
              radix16_max_blocks) {
            radix16_conf_[i].amount_of_blocks_per_kernel_ =
                (fft_length_ / (16 * 16 * 16)) / radix16_max_warps;
          } else {
            radix16_conf_[i].amount_of_blocks_per_kernel_ = radix16_max_blocks;
          }
        }
      }

      //Determine amount of kernels per radix 16 step
      if (radix16_conf_[i].amount_of_blocks_per_kernel_ < radix16_max_blocks) {
        radix16_conf_[i].amount_of_kernels_ = 1;
      } else {
        if (((fft_length_ / ((16 * 16 * 16) * radix16_max_warps)) %
            radix16_max_blocks) != 0) {
        std::cout << "Error! The total amount of blocks has to be devisable by"
                  << " radix16_max_blocks "
                  << "(i.e. (fft_length/(256*radix16_max_warps))"
                  << "%radix16_max_blocks == 0)."
                  << std::endl;
        return false;
        } else {
        radix16_conf_[i].amount_of_kernels_ =
            (fft_length_ / ((16 * 16 * 16) * radix16_max_warps)) /
            radix16_max_blocks;
        }
      }
      for(int i=0; i<radix16_conf_[i].amount_of_kernels_; i++){
        radix16_conf_[i].kernel_ids_.push_back(i);
      }
    }

    std::cout << "Creating Radix16 config successfull!" << std::endl;

    return true;
  }

  //Analogous to CreateDFTLaunchConfig but for the radix2 steps
  bool CreateRadix2LaunchConfig(int radix2_max_blocksize,
                                int radix2_max_blocks){
    radix2_conf_.resize(amount_of_radix_2_steps_);
    for(int i=0; i<amount_of_radix_2_steps_; i++){
      radix2_conf_[i].current_radix2_step_ = i;
      radix2_conf_[i].size_of_ffts_ =
          std::pow(16,amount_of_radix_16_steps_+1) * std::pow(2,i);
      radix2_conf_[i].amount_of_ffts_ =
          (fft_length_ / radix2_conf_[i].size_of_ffts_);

      //Determine blocksize
      if (radix2_conf_[i].size_of_ffts_ <= radix2_max_blocksize) {
        radix2_conf_[i].blocksize_ = radix2_conf_[i].size_of_ffts_;
      } else {
        radix2_conf_[i].blocksize_ = radix2_max_blocksize;
      }

      //Determine amount of blocks per kernel
      if (radix2_conf_[i].blocksize_ < radix2_max_blocksize) {
        radix2_conf_[i].amount_of_blocks_per_kernel_ = 1;
      } else {
        if ((radix2_conf_[i].size_of_ffts_ % radix2_max_blocksize) != 0) {
          std::cout << "Error! Total amount of threads has to be devisable by "
                    << "radix2_max_blocksize i.e. (16**amount_of_radix_16_"
                    << "steps_+1)%radix2_max_blocksize == 0"
                    << std::endl;
          return false;
        } else {
          if ((radix2_conf_[i].size_of_ffts_ / radix2_max_blocksize) <=
              radix2_max_blocks) {
            radix2_conf_[i].amount_of_blocks_per_kernel_ =
              radix2_conf_[i].size_of_ffts_ / radix2_max_blocksize;
          } else {
            radix2_conf_[i].amount_of_blocks_per_kernel_ = radix2_max_blocks;
          }
        }
      }

      //Determine amount of kernels
      if (radix2_conf_[i].amount_of_blocks_per_kernel_ < radix2_max_blocks) {
        radix2_conf_[i].amount_of_kernels_ = 1;
      } else {
        if (((radix2_conf_[i].size_of_ffts_ / radix2_max_blocksize) %
            radix2_max_blocks) != 0) {
          std::cout << "Error! Total amount of blocks has to be devisable by "
                    << "radix2_max_blocks i.e. (16**(amount_of_radix_16_"
                    << "steps_+1)/radix2_max_blocksize)%"
                    << "radix2_max_blocksize == 0"
                    << std::endl;
          return false;
        } else {
          radix2_conf_[i].amount_of_kernels_ =
              radix2_conf_[i].size_of_ffts_ /
              (radix2_max_blocksize * radix2_max_blocks);
        }
      }
      for(int j=0; j<radix2_conf_[i].amount_of_kernels_; j++){
        radix2_conf_[i].kernel_ids_.push_back(j);
        radix2_conf_[i].kernel_memory_offfset_.push_back(
            (fft_length_ / radix2_conf_[i].amount_of_kernels_) *
            radix2_conf_[i].kernel_ids_[j]);
      }
      radix2_conf_[i].amount_of_kernels_per_fft_ =
          radix2_conf_[i].amount_of_kernels_ /
          std::pow(2,amount_of_radix_2_steps_ -
                   radix2_conf_[i].current_radix2_step_ - 1);
      if (radix2_conf_[i].amount_of_kernels_per_fft_ == 0) {
        radix2_conf_[i].amount_of_kernels_per_fft_ = 1;
      }
    }

    std::cout << "Creating radix2 config successfull!" << std::endl;

    return true;
  }

  //Allocates needed memory for the fft on the device
  //TODO check alignment requirement
  bool AllocateDeviceMemory(){
    if (cudaMalloc(&dptr_data_, sizeof(__half2) * fft_length_)
        != cudaSuccess) {
      return false;
    }
    dptr_data_RE_ = (__half*)(dptr_data_);
    dptr_data_IM_ = (__half*)(dptr_data_) + fft_length_;

    if (cudaMalloc(&dptr_results_, sizeof(__half2) * fft_length_)
        != cudaSuccess) {
      return false;
    }
    dptr_results_RE_ = (__half*)(dptr_results_);
    dptr_results_IM_ = (__half*)(dptr_results_) + fft_length_;

    if (cudaMalloc(&dptr_dft_matrix_batch_RE_, sizeof(__half) * 16 * 16 * 16)
        != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(&dptr_dft_matrix_batch_IM_, sizeof(__half) * 16 * 16 * 16)
        != cudaSuccess) {
      return false;
    }

    std::cout << "Allocating device memory successfull!" << std::endl;

    return true;
  }

  //Creates amount_of_mem_copies_ node pairs, consisting of one cpy and one
  //transpose kernel that depends on that cpy. The node pairs are not dependend
  //on each other. Finnaly all node pairs are then packed into one single node.
  bool MakeCpyAndTransposeNode(){
    //Create memcpy_transpose_child_graph_
    if (cudaGraphCreate(&memcpy_transpose_child_graph_, 0) != cudaSuccess) {
      std::cout << "Error!  Initial child graph creation failed!"
                << std::endl;
      return false;
    }

    memcopies_.resize(transpose_conf_.amount_of_kernels_);
    transpose_kernels_.resize(transpose_conf_.amount_of_kernels_);
    transpose_kernel_params_.resize(transpose_conf_.amount_of_kernels_);

    //Set parameters needed for transpose kernel node addition
    for(int i=0; i<transpose_conf_.amount_of_kernels_; i++){
      std::vector<void*> tmp_args;

      tmp_args.push_back((void*)&dptr_data_);
      tmp_args.push_back((void*)&dptr_results_RE_);
      tmp_args.push_back((void*)&dptr_results_IM_);
      tmp_args.push_back((void*)&transpose_conf_.amount_of_kernels_);
      tmp_args.push_back((void*)&(transpose_conf_.kernel_ids_[i]));
      tmp_args.push_back((void*)&fft_length_);
      tmp_args.push_back((void*)&amount_of_radix_16_steps_);
      tmp_args.push_back((void*)&amount_of_radix_2_steps_);

      transpose_kernel_args_.push_back(tmp_args);

      transpose_kernel_params_[i].func = (void*)TransposeKernel;
      transpose_kernel_params_[i].gridDim =
          dim3(transpose_conf_.amount_of_blocks_per_kernel_, 1, 1);
      transpose_kernel_params_[i].blockDim =
          dim3(transpose_conf_.blocksize_, 1, 1);
      transpose_kernel_params_[i].sharedMemBytes = 0;
      transpose_kernel_params_[i].kernelParams =
          transpose_kernel_args_[i].data();
      transpose_kernel_params_[i].extra = nullptr;
    }

    //Adds pairs of memcpy and transpose kernel nodes to the
    //memcpy_transpose_child_graph_
    for(int i=0; i<transpose_conf_.amount_of_kernels_; i++){
      //The used memory per kenel is evenly and in order distributed to all
      //kernels. Thus the ptr to the memory for each kernel is
      //(xptr_data + memory_offset)
      int memory_offset =
          i * (fft_length_ / transpose_conf_.amount_of_kernels_);
      if (cudaGraphAddMemcpyNode1D(&(memcopies_[i]),
                                   memcpy_transpose_child_graph_,
                                   nullptr,
                                   0,
                                   (void*)(dptr_data_ + memory_offset),
                                   (void*)(hptr_data_ + memory_offset),
                                   sizeof(__half2) * (fft_length_ /
                                       transpose_conf_.amount_of_kernels_),
                                   cudaMemcpyHostToDevice)
          != cudaSuccess) {
        std::cout << "Error! Adding memcpy node failed!"
                  << std::endl;
        return false;
      }

      if (cudaGraphAddKernelNode(&(transpose_kernels_[i]),
                                 memcpy_transpose_child_graph_,
                                 &(memcopies_[i]), 1,
                                 &(transpose_kernel_params_[i]))
          != cudaSuccess) {
         std::cout << "Error! Adding transpose kernel node failed!"
                   << std::endl;
         std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
         return false;
      }

      //Set the data of the dft matrix batch that will bbe used in the dft and
      //r16 kernel. For a element of the 16x16 matrix a_nm = exp(-2PI*i*n*m/16)
      //16 consecutive matrices are needed to load one fragment in the dft and
      //r16 kernels.
      for(int i=0; i<16; i++){
        for(int j=0; j<16; j++){
          for(int k=0; k<16; k++){
            hptr_dft_matrix_batch_RE_[k + 16 * j + 16 * 16 * i] =
                __double2half(cos((-2*M_PI*j*k)/16.0));
            hptr_dft_matrix_batch_IM_[k + 16 * j + 16 * 16 * i] =
                __double2half(sin((-2*M_PI*j*k)/16.0));
          }
        }
      }

      if (cudaGraphAddMemcpyNode1D(&dft_matrix_batch_memcopy_,
                                   memcpy_transpose_child_graph_,
                                   nullptr,
                                   0,
                                   (void*)(dptr_dft_matrix_batch_RE_),
                                   (void*)(hptr_dft_matrix_batch_RE_),
                                   sizeof(__half) * 16 * 16 *16,
                                   cudaMemcpyHostToDevice)
          != cudaSuccess) {
        std::cout << "Error! Adding dft matrix memcpy node failed!"
                  << std::endl;
        return false;
      }
      if (cudaGraphAddMemcpyNode1D(&dft_matrix_batch_memcopy_,
                                   memcpy_transpose_child_graph_,
                                   nullptr,
                                   0,
                                   (void*)(dptr_dft_matrix_batch_IM_),
                                   (void*)(hptr_dft_matrix_batch_IM_),
                                   sizeof(__half) * 16 * 16 *16,
                                   cudaMemcpyHostToDevice)
          != cudaSuccess) {
        std::cout << "Error! Adding dft matrix memcpy node failed!"
                  << std::endl;
        return false;
      }
    }

    //Packing of memcpy_transpose_child_graph_ into node
    //memcpy_transpose_child_graph_node_, which gets added to the fft_graph_.
    if (cudaGraphAddChildGraphNode(&memcpy_transpose_child_graph_node_,
                                   fft_graph_, nullptr, 0,
                                   memcpy_transpose_child_graph_)
        != cudaSuccess) {
      std::cout << "Error! Addition of memcpy_transpose_child_graph_node_ to "
                << "graph failed."
                << std::endl;
      return false;
    }

    std::cout << "Adding of memcpy and transpose nodes successfull!" << std::endl;

    return true;
  }

  //Adds the dft kernel nodes to the graph that depend on the node
  //memcpy_transpose_child_graph_node_
  bool MakeDFTNodes(){
    dft_kernels_.resize(dft_conf_.amount_of_kernels_);
    dft_kernel_params_.resize(dft_conf_.amount_of_kernels_);

    //Set parameters needed for dft kernel node additions
    for(int i=0; i<dft_conf_.amount_of_kernels_; i++){
      std::vector<void*> tmp_args;

      //The input for the dft step is created by the transpose kernels which
      //stored their results in dptr_results_
      tmp_args.push_back((void*)&dptr_results_RE_);
      tmp_args.push_back((void*)&dptr_results_IM_);

      tmp_args.push_back((void*)&dptr_data_RE_);
      tmp_args.push_back((void*)&dptr_data_IM_);
      tmp_args.push_back((void*)&hptr_dft_matrix_batch_RE_);
      tmp_args.push_back((void*)&hptr_dft_matrix_batch_IM_);
      tmp_args.push_back((void*)&dft_conf_.amount_of_kernels_);
      tmp_args.push_back((void*)&(dft_conf_.kernel_ids_[i]));
      tmp_args.push_back((void*)&fft_length_);

      dft_kernel_args_.push_back(tmp_args);

      dft_kernel_params_[i].func = (void*)DFTKernel;
      dft_kernel_params_[i].gridDim =
          dim3(dft_conf_.amount_of_blocks_per_kernel_, 1, 1);
      dft_kernel_params_[i].blockDim =
          dim3(dft_conf_.blocksize_, 1, 1);
      dft_kernel_params_[i].sharedMemBytes = 0;
      dft_kernel_params_[i].kernelParams = dft_kernel_args_[i].data();
      dft_kernel_params_[i].extra = nullptr;
    }

    //Adds the dft kernel nodes to the graph which depend on the
    //memcpy_transpose_child_graph_node_ node
    for(int i=0; i<dft_conf_.amount_of_kernels_; i++){
      if (cudaGraphAddKernelNode(&(dft_kernels_[i]), fft_graph_,
                                 &memcpy_transpose_child_graph_node_, 1,
                                 &(dft_kernel_params_[i])) != cudaSuccess) {
         std::cout << "Error! Adding dft kernel node failed!"
                   << std::endl;
         return false;
      }
    }

    std::cout << "Adding of dft nodes successfull!" << std::endl;

    return true;
  }

  //Prepares parameter objects which are used for the creation of the radix16
  //nodes
  void PrepareRadix16Parameters(){
    for(int i=0; i<amount_of_radix_16_steps_; i++){
      std::vector<cudaGraphNode_t> tmp_r16_kernels;
      std::vector<cudaKernelNodeParams> tmp_r16_kernel_params;
      tmp_r16_kernels.resize(radix16_conf_[i].amount_of_kernels_);
      tmp_r16_kernel_params.resize(radix16_conf_[i].amount_of_kernels_);
      radix16_kernels_.push_back(tmp_r16_kernels);
      radix16_kernel_params_.push_back(tmp_r16_kernel_params);
    }

    //Prepare and store kernel arguments
    for(int j=0; j<amount_of_radix_16_steps_; j++){
      std::vector<std::vector<void*>> tmp_args_vector;
      for(int i=0; i<radix16_conf_[j].amount_of_kernels_; i++){
        std::vector<void*> tmp_args;

        //For even indecies the inputdata is located in the data arrays and
        //stored in the results arrays and vice versa for odd indecies.
        if ((j%2) != 0) {
          tmp_args.push_back((void*)&dptr_results_RE_);
          tmp_args.push_back((void*)&dptr_results_IM_);
          tmp_args.push_back((void*)&dptr_data_RE_);
          tmp_args.push_back((void*)&dptr_data_IM_);
        }  else {
          tmp_args.push_back((void*)&dptr_data_RE_);
          tmp_args.push_back((void*)&dptr_data_IM_);
          tmp_args.push_back((void*)&dptr_results_RE_);
          tmp_args.push_back((void*)&dptr_results_IM_);
        }

        tmp_args.push_back((void*)&radix16_conf_[j].amount_of_kernels_);
        tmp_args.push_back((void*)&(radix16_conf_[j].kernel_ids_[i]));
        tmp_args.push_back((void*)&fft_length_);
        tmp_args.push_back((void*)&(radix16_conf_[j].current_radix16_step_));

        tmp_args_vector.push_back(tmp_args);
      }
      radix16_kernel_args_.push_back(tmp_args_vector);
    }

    //Set parameters needed for radix16 kernel node additions
    for(int j=0; j<amount_of_radix_16_steps_; j++){
      for(int i=0; i<radix16_conf_[j].amount_of_kernels_; i++){

        radix16_kernel_params_[j][i].func = (void*)Radix16Kernel;
        radix16_kernel_params_[j][i].gridDim =
            dim3(radix16_conf_[j].amount_of_blocks_per_kernel_, 1, 1);
        radix16_kernel_params_[j][i].blockDim =
            dim3(radix16_conf_[j].blocksize_, 1, 1);
        radix16_kernel_params_[j][i].sharedMemBytes =
            sizeof(__half) * dft_conf_.amount_of_warps_per_block_ * 8192;
        radix16_kernel_params_[j][i].kernelParams =
            radix16_kernel_args_[j][i].data();
        radix16_kernel_params_[j][i].extra = nullptr;
      }
    }
  }

  //Prepares parameter objects which are used for the creation of the radix2_dependencies
  //nodes
  void PrepareRadix2Parameters(){
    for(int i=0; i<amount_of_radix_2_steps_; i++){
      std::vector<cudaGraphNode_t> tmp_r2_kernels;
      std::vector<cudaKernelNodeParams> tmp_r2_kernel_params;
      tmp_r2_kernels.resize(radix2_conf_[i].amount_of_kernels_);
      tmp_r2_kernel_params.resize(radix2_conf_[i].amount_of_kernels_);
      radix2_kernels_.push_back(tmp_r2_kernels);
      radix2_kernel_params_.push_back(tmp_r2_kernel_params);
    }

    //Prepare and store kernel arguments
    for(int j=0; j<amount_of_radix_2_steps_; j++){
      std::vector<std::vector<void*>> tmp_args_vector;
      for(int i=0; i<radix2_conf_[j].amount_of_kernels_; i++){
        std::vector<void*> tmp_args;

        //Depending on the amount of previously performed radix16 and radix2
        //steps the input data is either located in the data or result arrays
        if (((amount_of_radix_16_steps_ + i) % 2) != 0) {
          tmp_args.push_back((void*)&dptr_results_RE_);
          tmp_args.push_back((void*)&dptr_results_IM_);
          tmp_args.push_back((void*)&dptr_data_RE_);
          tmp_args.push_back((void*)&dptr_data_IM_);
        }  else {
          tmp_args.push_back((void*)&dptr_data_RE_);
          tmp_args.push_back((void*)&dptr_data_IM_);
          tmp_args.push_back((void*)&dptr_results_RE_);
          tmp_args.push_back((void*)&dptr_results_IM_);
        }

        tmp_args.push_back((void*)&(radix2_conf_[j].kernel_memory_offfset_[i]));
        tmp_args.push_back((void*)&(radix2_conf_[j].size_of_ffts_));
        tmp_args.push_back((void*)&(radix2_conf_[j].amount_of_kernels_per_fft_));
        tmp_args.push_back((void*)&(radix16_conf_[j].current_radix16_step_));

        tmp_args_vector.push_back(tmp_args);
      }
      radix2_kernel_args_.push_back(tmp_args_vector);
    }

    //Set parameters needed for radix2 kernel node additions
    for(int j=0; j<amount_of_radix_2_steps_; j++){
      for(int i=0; i<radix2_conf_[j].amount_of_kernels_; i++){

        radix2_kernel_params_[j][i].func = (void*)Radix2Kernel;
        radix2_kernel_params_[j][i].gridDim =
            dim3(radix2_conf_[j].amount_of_blocks_per_kernel_, 1, 1);
        radix2_kernel_params_[j][i].blockDim =
            dim3(radix2_conf_[j].blocksize_, 1, 1);
        radix2_kernel_params_[j][i].sharedMemBytes = 0;
        radix2_kernel_params_[j][i].kernelParams =
            radix2_kernel_args_[j][i].data();
        radix2_kernel_params_[j][i].extra = nullptr;
      }
    }
  }

  std::vector<std::vector<int>> FindOverlappingNodes(
      std::vector<int> previous_step_first_proccessed_element,
      std::vector<int> previous_step_last_proccessed_element,
      std::vector<int> this_r16_first_needed_element,
      std::vector<int> this_r16_last_needed_element){
    std::vector<std::vector<int>> this_step_dependency_ids;
    //Loop over all nodes of the current step
    for(int j=0; j<static_cast<int>(this_r16_last_needed_element.size()); j++){
      std::vector<int> single_node_depenency_ids;
      int first_needed_node;
      int last_needed_node;
      //Loop forwards over all nodes of previous step to find first
      //overlapping node
      for(int k=0;
          k<static_cast<int>(previous_step_last_proccessed_element.size());
          k++){
        //Overlap with left boundry
        if ((this_r16_first_needed_element[j] >=
             previous_step_first_proccessed_element[k]) &&
            (this_r16_first_needed_element[j] <=
             previous_step_last_proccessed_element[k])) {
          first_needed_node = k;
          break;
        }
        //Overlap with right boundry
        if ((this_r16_last_needed_element[j] >=
             previous_step_first_proccessed_element[k]) &&
            (this_r16_last_needed_element[j] <=
             previous_step_last_proccessed_element[k])) {
          first_needed_node = k;
          break;
        }
        //Intervall enclosed
        if ((this_r16_first_needed_element[j] <=
             previous_step_first_proccessed_element[k]) &&
            (this_r16_last_needed_element[j] >=
             previous_step_last_proccessed_element[k])) {
          first_needed_node = k;
          break;
        }
      }
      //Loop backwards over all nodes of previous step to find last
      //overlapping node
      for(int k=previous_step_last_proccessed_element.size()-1; k>-1; k--){
        //Overlap with left boundry
        if ((this_r16_first_needed_element[j] >=
             previous_step_first_proccessed_element[k]) &&
            (this_r16_first_needed_element[j] <=
             previous_step_last_proccessed_element[k])) {
          last_needed_node = k;
          break;
        }
        //Overlap with right boundry
        if ((this_r16_last_needed_element[j] >=
             previous_step_first_proccessed_element[k]) &&
            (this_r16_last_needed_element[j] <=
             previous_step_last_proccessed_element[k])) {
          last_needed_node = k;
          break;
        }
        //Intervall enclosed
        if ((this_r16_first_needed_element[j] <=
             previous_step_first_proccessed_element[k]) &&
            (this_r16_last_needed_element[j] >=
             previous_step_last_proccessed_element[k])) {
          last_needed_node = k;
          break;
        }
      }

      single_node_depenency_ids.push_back(first_needed_node);
      single_node_depenency_ids.push_back(last_needed_node);
      this_step_dependency_ids.push_back(single_node_depenency_ids);
    }
    return this_step_dependency_ids;
  }

  //TODO break up into smaller functions
  bool MakeRadixNodes(bool one_or_more_r16, bool one_or_more_r2){
    //If there are radix16 steps to be done, add the according nodes to the
    //graph
    if (one_or_more_r16) {
      PrepareRadix16Parameters();

      std::cout << "Prepareing r16 para successfull!" << std::endl;

      //Determine dependendcies of every radix 16 kernel to the previous kernels
      //Each std::vector<int> contains the id of first and last nodes of the
      //previous step that the current node depends on
      std::vector<std::vector<std::vector<int>>> radix16_dependencies;
      //Loop over all radix16 steps
      for(int i=0; i<amount_of_radix_16_steps_; i++){
        //These vectors are used to Determine the dependendcies between the
        //kernel nodes of two folloing steps (e.g. dft -> radix16 nr.1 or
        //radix16 nr.i -> radix16 nr.i+1)
        //The dft and radix kernels operate on a subsequent linear pieces of
        //memory whos length is fft_length_/amount_of_kernels_.
        //A kernel depends on a kernel of a previous step if its input memory
        //overlaps with the output memory of such a kernel.
        std::vector<int> previous_step_first_proccessed_element;
        std::vector<int> previous_step_last_proccessed_element;
        std::vector<int> this_r16_first_needed_element;
        std::vector<int> this_r16_last_needed_element;
        if (i == 0){
          for(int k=0; k<dft_conf_.amount_of_kernels_; k++){
            previous_step_first_proccessed_element.push_back(k*(fft_length_ /
                dft_conf_.amount_of_kernels_));
            previous_step_last_proccessed_element.push_back((k+1)*(fft_length_ /
                dft_conf_.amount_of_kernels_));
          }
        } else {
          for(int k=0; k<radix16_conf_[i-1].amount_of_kernels_; k++){
            previous_step_first_proccessed_element.push_back(k*(fft_length_ /
                radix16_conf_[i-1].amount_of_kernels_));
            previous_step_last_proccessed_element.push_back((k+1)*(fft_length_ /
                radix16_conf_[i-1].amount_of_kernels_));
          }
        }

        for(int k=0; k<radix16_conf_[i].amount_of_kernels_; k++){
          this_r16_first_needed_element.push_back(k*(fft_length_ /
              radix16_conf_[i].amount_of_kernels_));
          this_r16_last_needed_element.push_back((k+1)*(fft_length_ /
              radix16_conf_[i].amount_of_kernels_));
        }

        std::cout << "Compute Overlap." << std::endl;

        std::vector<std::vector<int>> tmp_r16_dependencies =
            FindOverlappingNodes(previous_step_first_proccessed_element,
                                 previous_step_last_proccessed_element,
                                 this_r16_first_needed_element,
                                 this_r16_last_needed_element);
        radix16_dependencies.push_back(tmp_r16_dependencies);
      }

      std::cout << "Computing r16 overlapp successfull!" << std::endl;

      //Add the radix 16 nodes
      for(int i=0; i<amount_of_radix_16_steps_; i++){
        for(int j=0; j<radix16_conf_[i].amount_of_kernels_; j++){
          if (i == 0) { //dft step is previous step
            if (cudaGraphAddKernelNode(
                    &(radix16_kernels_[i][j]), fft_graph_,
                    &(dft_kernels_[radix16_dependencies[i][j][0]]),
                    radix16_dependencies[i][j][1]-radix16_dependencies[i][j][0],
                    &(radix16_kernel_params_[i][j]))
                != cudaSuccess) {
               std::cout << "Error! Adding radix16 kernel node failed!"
                         << std::endl;
               return false;
            }
          } else {  //radix 16 step is previous step
            if (cudaGraphAddKernelNode(
                    &(radix16_kernels_[i][j]), fft_graph_,
                    &(radix16_kernels_[i-1][radix16_dependencies[i][j][0]]),
                    radix16_dependencies[i][j][1]-radix16_dependencies[i][j][0],
                    &(radix16_kernel_params_[i][j]))
                != cudaSuccess) {
               std::cout << "Error! Adding radix16 kernel node failed!"
                         << std::endl;
               return false;
            }
          }
        }
      }
    }
    std::cout << "Adding r16 nodes successfull!" << std::endl;

    //If there are radix2 steps to be done, add the according nodes to the
    //graph
    if (one_or_more_r2) {
      PrepareRadix2Parameters();

      std::cout << "Prepareing r2 para successfull!" << std::endl;

      //Determine dependendcies of every radix 2 kernel to the previous kernels
      //Each std::vector<int> contains the id of first and last nodes of the
      //previous step that the current node depends on
      std::vector<std::vector<std::vector<int>>> radix2_dependencies;
      //Loop over all radix16 steps
      for(int i=0; i<amount_of_radix_2_steps_; i++){
        //These vectors are used to Determine the dependendcies between the
        //kernel nodes of two following steps (e.g. dft -> radix2 nr.1 or
        //radix16 nr.n -> radix2 nr.1 or radix2 nr.i -> radix2 nr.i+1)
        //The dft and radix kernels operate on a subsequent linear pieces of
        //memory whos length is fft_length_/amount_of_kernels_.
        //A kernel depends on a kernel of a previous step if its input memory
        //overlaps with the output memory of such a kernel.
        std::vector<int> previous_step_first_proccessed_element;
        std::vector<int> previous_step_last_proccessed_element;
        std::vector<int> this_r2_first_needed_element;
        std::vector<int> this_r2_last_needed_element;
        if (i == 0){
          if (amount_of_radix_16_steps_ != 0) {
            for(int k=0;
                k<radix16_conf_[amount_of_radix_16_steps_-1].amount_of_kernels_;
                k++){
              previous_step_first_proccessed_element.push_back(k*(fft_length_ /
                radix16_conf_[amount_of_radix_16_steps_-1].amount_of_kernels_));
              previous_step_last_proccessed_element.push_back((k+1)*
                (fft_length_ /
                radix16_conf_[amount_of_radix_16_steps_-1].amount_of_kernels_));
            }
          } else {
            for(int k=0; k<dft_conf_.amount_of_kernels_; k++){
              previous_step_first_proccessed_element.push_back(k*(fft_length_ /
                dft_conf_.amount_of_kernels_));
              previous_step_last_proccessed_element.push_back((k+1)*
                (fft_length_ /
                dft_conf_.amount_of_kernels_));
            }
          }
        } else {
          for(int k=0; k<radix2_conf_[i-1].amount_of_kernels_; k++){
            previous_step_first_proccessed_element.push_back(k*(fft_length_ /
                radix2_conf_[i-1].amount_of_kernels_));
            previous_step_last_proccessed_element.push_back((k+1)*(fft_length_ /
                radix2_conf_[i-1].amount_of_kernels_));
          }
        }

        for(int k=0; k<radix2_conf_[i].amount_of_kernels_; k++){
          this_r2_first_needed_element.push_back(k*(fft_length_ /
              radix2_conf_[i].amount_of_kernels_));
          this_r2_last_needed_element.push_back((k+1)*(fft_length_ /
              radix2_conf_[i].amount_of_kernels_));
        }

        std::vector<std::vector<int>> tmp_r2_dependencies =
            FindOverlappingNodes(previous_step_first_proccessed_element,
                                 previous_step_last_proccessed_element,
                                 this_r2_first_needed_element,
                                 this_r2_last_needed_element);
        radix2_dependencies.push_back(tmp_r2_dependencies);
      }

      std::cout << "Computing r16 overlapp successfull!" << std::endl;

      //Add the radix 2 nodes
      for(int i=0; i<amount_of_radix_2_steps_; i++){
        for(int j=0; j<radix2_conf_[i].amount_of_kernels_; j++){
          if (i == 0) {
            if (amount_of_radix_16_steps_ == 0) { //dft step is previous step
              if (cudaGraphAddKernelNode(
                      &(radix2_kernels_[i][j]), fft_graph_,
                      &(dft_kernels_[radix2_dependencies[i][j][0]]),
                      radix2_dependencies[i][j][0] -
                      radix2_dependencies[i][j][1] + 1,
                      &(radix2_kernel_params_[i][j]))
                  != cudaSuccess) {
                 std::cout << "Error! Adding radix2 kernel node failed!"
                           << std::endl;
                 return false;
              }
            } else { //radix16 is previous step
              if (cudaGraphAddKernelNode(
                      &(radix2_kernels_[i][j]), fft_graph_,
                      &(radix16_kernels_[amount_of_radix_16_steps_-1]
                                        [radix2_dependencies[i][j][0]]),
                      radix2_dependencies[i][j][0] -
                      radix2_dependencies[i][j][1] + 1,
                      &(radix2_kernel_params_[i][j]))
                  != cudaSuccess) {
                 std::cout << "Error! Adding radix2 kernel node failed!"
                           << std::endl;
                 return false;
              }
            }
          } else {  //radix2 step is previous step
            if (cudaGraphAddKernelNode(
                    &(radix2_kernels_[i][j]), fft_graph_,
                    &(radix2_kernels_[i-1][radix2_dependencies[i][j][0]]),
                    radix2_dependencies[i][j][0] -
                    radix2_dependencies[i][j][1] + 1,
                    &(radix2_kernel_params_[i][j]))
                != cudaSuccess) {
               std::cout << "Error! Adding radix2 kernel node failed!"
                         << std::endl;
               return false;
            }
          }
        }
      }
    }
    std::cout << "Adding r2 nodes successfull!" << std::endl;

    std::cout << "Adding of radix nodes successfull!" << std::endl;

    return true;
  }

  //Allocates the needed memory on the device (roughly 2xinput data size),
  //creates the fft_graph_ and adds all needed nodes to it.
  bool CreateGraph(){
    if (!AllocateDeviceMemory()) {
      std::cout << "Error! Memory allocation on device failed!"
                << std::endl;
      return false;
    }

    if (cudaGraphCreate(&fft_graph_, 0) != cudaSuccess) {
      std::cout << "Error! Initial graph creation failed!"
                << std::endl;
      return false;
    }

    if (!MakeCpyAndTransposeNode()) {
      std::cout << "Error! Adding cpy and transpose nodes failed."
                << std::endl;
      return false;
    }

    if (!MakeDFTNodes()) {
      std::cout << "Error! Adding DFT nodes failed."
                << std::endl;
      return false;
    }

    bool one_or_more_r16;
    bool one_or_more_r2;

    if (amount_of_radix_16_steps_ == 0) {
      one_or_more_r16 = false;
    } else {
      one_or_more_r16 = true;
    }

    if (amount_of_radix_2_steps_ == 0) {
      one_or_more_r2 = false;
    } else {
      one_or_more_r2 = true;
    }

    if (!MakeRadixNodes(one_or_more_r16, one_or_more_r2)) {
      std::cout << "Error! Adding Radix nodes failed."
                << std::endl;
      return false;
    }

    if (cudaGraphInstantiate(&executable_fft_graph_, fft_graph_, nullptr,
                             nullptr, 0) != cudaSuccess) {
      std::cout << "Error! Creating executable graph failed."
                << std::endl;
      return false;
    }

    graph_is_valid_ = true;
    return true;
  }
};
