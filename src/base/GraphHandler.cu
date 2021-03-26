//Contains mostly the GraphHandler class which is used to create a graph of
//memory copies and kernel launches needed to perform the FFT of a specified
//input and execute it.
#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

bool IsPowerOf2(int x) {
  if (x=0){
    return false;
  }
  return ((n & (n - 1)) == 0);
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

//Contains the needed information for the creation of the kernels of the base
//layer DFT step of the FFT
struct DFTLaunchConfig {
  int amount_of_ffts_;
  int amount_of_warps_per_block_;
  int blocksize_;
  int amount_of_blocks_per_kernel_;
  int amount_of_matrices_per_warp_;
  int amount_of_kernels_;
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
};

//This class is used to perform the fft of a given input.
//Calling the constructor creates a graph of kernels that can then be executed
//via ExecuteGraph() to compute the fft.
//The main parameter for the constructor are the fft lenght and a ptr to a
//__half2 array which holds the complex fp16 input data.
//This FFT implementation utilizes the radix16 and radix2 variants of the
//Cooley-Tukey algorithm and implements them on the GPU, with the radix16 part
//of the calculation being accelerated by tensor cores. Due to the used
//algorithms the input size is restricted to powers of 2. Expressing the input
//size N=2**M=2**K * 16**L, where K is the minimal value for a given N,
//comparative performance is best for large values of L and small fractions K/L,
//due to the fact that only the radix16 part of the algorithm is accelerated by
//tensor cores.
//The template parameter radix2_loop_length and the other parameter of the
//constructor are performance parameter. For more detail on them and some
//restrictions on them rising from the fft lenght referre to the functions
//Create...LaunchConfig().
template<int radix2_loop_length>
class GraphHandler {
private:
  int fft_length_;
  int amount_of_radix_16_steps_; //i.e. floor(log16(fft_length))-1
  //i.e. log2(fft_length) - 4*floor(log16(fft_length))
  int amount_of_radix_2_steps_;
  //used to signal whether the graph creation was successfull
  bool graph_is_valid_;
  //number of memcpy the data transfer to the gpu is split into
  int amount_of_mem_copies_;
  int data_batch_size_; //in amount of __half2
  DFTLaunchConfig dft_conf_;
  std::vector<Radix16LaunchConfig> radix16_conf_;
  std::vector<Radix2LaunchConfig> radix2_conf_;
public:
  GraphHandler(int fft_length, std::unique_ptr<__half2> data,
               int amount_host_to_device_memcopies, int dft_max_warps,
               int dft_max_blocks, int radix16_max_warps,
               int radix16_max_blocks, int radix2_max_blocksize,
               int radix2_max_blocks)
      : fft_length_(fft_length), graph_is_ready_(false),
        amount_of_mem_copies_(amount_host_to_device_memcopies) {
    data_batch_size_ = fft_length_ / amount_of_mem_copies_;

    //Parse input length and set up launch configs and if successfull create
    //graph
    if (ParseInputLength() &&
        (CreateDFTLaunchConfig(dft_max_warps, dft_max_blocks) &&
         CreateRadix16LaunchConfig(radix16_max_warps, radix16_max_blocks)
         && CreateRadix2LaunchConfig(radix2_max_blocksize, radix2_max_blocks))) {
      CreateGraph();
    } else {
      std::cout << "Graph could not be created" << std::endl;
    }
  }

  //Determines the amount of radix16 and radix2 steps.
  //Since the base dft step works on 16 points and uses tensor cores which
  //require, for the current wwma api, batches of size 16, the intput size has
  //to be devisable by 256 (thus also fft_length >= 256).
  bool ParseInputLength(){
    if (((fft_length_ % 256) != 0) || !(IsPowerOf2(fft_length_))) {
      std::cout << "Input size is NOT a power of 2 or devisable by 256! This "
                << "algorithm only supports inputsizes that are larger than 256"
                << " and powers of 2."
                << std::endl;
      return false;
    } else {
      int log2 = ExactLog2(fft_length_);
      amount_of_radix_16_steps_ = (log2 / 4) - 1;
      amount_of_radix_2_steps_ = log2 % 4;
      std::cout << "FFT lenght: " << fft_length_ << " = 2**" << log2 << " = 2**"
                << amount_of_radix_2_steps_ << "*16**" << log2 / 4 << std::endl
                << "Amount of DFT steps: 1" << std::endl;
                << "Amount of radix 16 steps: " << amount_of_radix_16_steps_
                << std::endl
                << "Amount of radix 2 steps: " << amount_of_radix_2_steps_
                << std::endl;
      return true;
    }
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
  //on one SM but will cause register spilling if increased to high. Increasing
  //dft_max_blocks will cause better ocupancy on the total amount of SMs per GPU
  //but decreasing it will cause the launch of more kernels which will make it
  //possible to execute further steps (i.e. the radix steps) on data produced by
  //kernels that have already finished.
  //More information about the performance implications of launch configurations
  //can be found in the nvidia proframming guide for cuda.
  bool CreateDFTLaunchConfig(int dft_max_warps, int dft_max_blocks){
    dft_conf_.amount_of_ffts_ = fft_length_ / 16;
    dft_conf_.amount_of_matrices_per_warp_ = 16; //required by current wwma api

    //Determines the amount of warps per block and thus also the blocksize
    if ((dft_conf_.amount_of_ffts_ / 16) <= dft_max_warps) {
      dft_conf_.amount_of_warps_per_block_ = dft_conf_.amount_of_ffts_ / 16;
    } else {
      dft_conf_.amount_of_warps_per_block_ = dft_max_warps;
    }
    dft_conf_.blocksize_ = dft_conf_.amount_of_warps_per_block_ * 32;

    //Determines the amount of blocks per kernel
    if (dft_conf_.amount_of_warps_per_block_ < dft_max_warps){
      dft_conf_.amount_of_blocks_per_kernel_ = 1;
    } else {
      if (((dft_conf_.amount_of_ffts_ / 16) % dft_max_warps) != 0) {
        std::cout << "Error! The total amount of warps has to be devisable by "
                  << "dft_max_warps (i.e. (fft_length/256)%dft_max_warps == 0)."
                  << std::endl;
        return false;
      } else {
        if (((dft_conf_.amount_of_ffts_ / 16) / dft_max_warps) <=
            dft_max_blocks) {
          dft_conf_.amount_of_blocks_per_kernel_ =
              (dft_conf_.amount_of_ffts_ / 16) / dft_max_warps;
        } else {
          dft_conf_.amount_of_blocks_per_kernel_ = dft_max_blocks;
        }
      }
    }

    //determines the amount of dft kernels to be launched
    if (dft_conf_.amount_of_blocks_per_kernel_ < dft_max_blocks) {
      dft_conf_.amount_of_kernels_ = 1;
    } else {
      if (((dft_conf_.amount_of_ffts_ / (16 * dft_max_warps)) % dft_max_blocks)
          != 0) {
        std::cout << "Error! The total amount of blocks has to be devisable by "
                  << "dft_max_blocks (i.e. "
                  << "(fft_length/(256*dft_max_warps))%dft_max_blocks == 0)."
                  << std::endl;
        return false;
      } else {
        dft_conf_.amount_of_kernels_ =
            (dft_conf_.amount_of_ffts_ / (16 * dft_max_warps)) % dft_max_blocks;
      }
    }
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
      if ((fft_length_ / 256) <= radix16_max_warps) {
        radix16_conf_[i].amount_of_warps_per_block_ = fft_length_ / 256;
      } else {
        radix16_conf_[i].amount_of_warps_ = radix16_max_warps;
      }
      radix16_conf_[i].blocksize_ =
          radix16_conf_[i].amount_of_warps_per_block_ * 32;

      //Determine amount of blocks per kernel
      if (radix16_conf_[i].amount_of_warps_per_block_ < radix16_max_warps) {
        radix16_conf_[i].amount_of_blocks_per_kernel_ = 1;
      } else {
        if (((fft_length_ / 256) % radix16_max_warps) != 0) {
          std::cout << "Error! The total amount of warps has to be devisable by"
                    << " radix16_max_warps "
                    << "(i.e. (fft_length/256)%radix16_max_warps == 0)."
                    << std::endl;
          return false;
        } else {
          if (((fft_length_ / 256) / radix16_max_warps) <= radix16_max_blocks) {
            radix16_conf_[i].amount_of_blocks_per_kernel_ =
                (fft_length_ / 256) / radix16_max_warps;
          } else {
            radix16_conf_[i].amount_of_blocks_per_kernel_ = radix16_max_blocks;
          }
        }
      }

      //Determine amount of kernels per radix 16 step
      if (radix16_conf_[i].amount_of_blocks_per_kernel_ < radix16_max_blocks) {
        radix16_conf_[i].amount_of_kernels_ = 1;
      } else {
        if (((fft_length_ / (256 * radix16_max_warps)) % radix16_max_blocks)
            != 0) {
        std::cout << "Error! The total amount of blocks has to be devisable by"
                  << " radix16_max_blocks "
                  << "(i.e. (fft_length/(256*radix16_max_warps))
                  << "%radix16_max_blocks == 0)."
                  << std::endl;
        return false;
        } else {
        radix16_conf_[i].amount_of_kernels_ =
            (fft_length_ / (256 * radix16_max_warps)) / radix16_max_blocks;
        }
      }

    }
    return true;
  }

  //Anlalogous to CreateDFTLaunchConfig but for the radix2 steps
  //The template parameter radix2_loop_length determines how many calculations
  //are done per thread for one radix2 step of two ffts. More details in FFT2.cu
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
      if ((radix2_conf_[i].size_of_ffts_ % radix2_loop_length) != 0) {
        std::cout << "Error! Size of fft has to devisable by radix2_loop_length"
                  << " i.e. 16**(amount_of_radix_16_steps_+1)%"
                  << "radix2_loop_length == 0."
                  << std::endl;
        return false;
      } else {
        if ((radix2_conf_[i].size_of_ffts_ / radix2_loop_length) <=
            radix2_max_blocksize) {
          radix2_conf_[i].blocksize_ =
              radix2_conf_[i].size_of_ffts_ / radix2_loop_length;
        } else {
          radix2_conf_[i].blocksize_ = radix2_max_blocksize;
        }
      }

      //Determine amount of blocks per kernel
      if (radix2_conf_[i].blocksize_ < radix2_max_blocksize) {
        radix2_conf_[i].amount_of_blocks_per_kernel_ = 1;
      } else {
        if ((radix2_conf_[i].size_of_ffts_ / radix2_loop_length) %
            radix2_max_blocksize) != 0) {
          std::cout << "Error! Total amount of threads has to be devisable by "
                    << "radix2_max_blocksize i.e. (16**(amount_of_radix_16_"
                    << "steps_+1)/radix2_loop_length)%radix2_max_blocksize == 0"
                    << std::endl;
          return false;
        } else {
          if (((radix2_conf_[i].size_of_ffts_ / radix2_loop_length) /
               radix2_max_blocksize) <= radix2_max_blocks) {
            radix2_conf_[i].amount_of_blocks_per_kernel_ =
              (radix2_conf_[i].size_of_ffts_ / radix2_loop_length) /
              radix2_max_blocksize;
          } else {
            radix2_conf_[i].amount_of_blocks_per_kernel_ = radix2_max_blocks;
          }
        }
      }

      //Determine amount of kernels
      if (radix2_conf_[i].amount_of_blocks_per_kernel_ < radix2_max_blocks) {
        radix2_conf_[i].amount_of_kernels_ = 1;
      } else {
        if (((radix2_conf_[i].size_of_ffts_ /
              (radix2_loop_length * radix2_max_blocksize))
             % radix2_max_blocks) != 0) {
          std::cout << "Error! Total amount of blocks has to be devisable by "
                    << "radix2_max_blocks i.e. (16**(amount_of_radix_16_"
                    << "steps_+1)/(radix2_loop_length*radix2_max_blocksize))%"
                    << "radix2_max_blocksize == 0"
                    << std::endl;
          return false;
        } else {
          radix2_conf_[i].amount_of_kernels_ =
              radix2_conf_[i].size_of_ffts_ /
              (radix2_loop_length * radix2_max_blocksize * radix2_max_blocks);
        }
      }

    }
    return true;
  }


  void CreateGraph()
  void ExecuteGraph()
};
