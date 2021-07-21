//Used to test functonality of Multti GPU capabilities
#include <iostream>
#include <string>

#include "FFTTestMultiGPU.cu"

int main(){
  if (!(TestMultiGPU(16*16, 4, 4, 0.1, 0.1))){
    std::cout << "Multi GPU test on 4 devices, with 4 ffts each of length 16*16"
              << " failed!" << std::endl;
    return false;
  }
  if (!(TestMultiGPU(16*16*16*2*2*2, 4, 4, 0.1, 0.1))){
    std::cout << "Multi GPU test on 4 devices, with 4 ffts each of length 16*16"
              << "*16*2*2*2 failed!" << std::endl;
    return false;
  }

  std::cout << "All tests passed!" <<std::endl;

  return true;
}
