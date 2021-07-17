//Used to test functonality
#include <iostream>
#include <string>

#include "TransposeTest.cu"
#include "DFTMatrixTest.cu"
#include "DFTTest.cu"

int main(){
  if (!transpose16_test()){
    std::cout << "Transpose 16*16*16 test failed!" << std::endl;
    return false;
  }
  if (!transpose16_2_test()){
    std::cout << "Transpose 16*16*16*2*2 test failed!" << std::endl;
    return false;
  }

  if (!dft_matrix16_test()){
    std::cout << "DFT matrix test 16*16 failed!" << std::endl;
    return false;
  }
  if (!dft_matrix16_2_test()){
    std::cout << "DFT matrix test 16*16*16*16*2 failed!" << std::endl;
    return false;
  }

  if (!dft_0_test()){
    std::cout << "DFT kernel test failed for 16*16*16 with value 0!"
              << std::endl;
    return false;
  }
  if (!dft_sin_test_16()){
    std::cout << "DFT kernel test failed for 16*16*16 with sin(x) x [0:2*PI]!"
              << std::endl;
    return false;
  }
  if (!dft_sin_test_2()){
    std::cout << "DFT kernel test failed for 16*16*16*16*2 with sin(x) x"
              << " [0:2*PI]!"
              << std::endl;
    return false;
  }

  if (!full_test_16()){
    std::cout << "FFT test for a length of 16*16 failed." << std::endl;
    return false;
  }

  std::cout << "All tests passed!" <<std::endl;

  return true;
}
