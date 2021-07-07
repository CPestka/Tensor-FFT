//Used to test functonality
#include <iostream>
#include <string>

#include "TransposeTest.cu"
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

  if (!dft16_test()){
    std::cout << "DFT kernel test failed for 16*16*16!" << std::endl;
    return false;
  }

  /*
  if (!dft_16_2_test()){
    std::cout << "DFT kernel test failed for 16*16*16*2*2!" << std::endl;
    return;
  }

  if (!r16_16_test()){
    std::cout << "R16 kernel test failed for 16*16*16!" << std::endl;
    return;
  }
  if (!r16_16_2_test()){
    std::cout << "R16 kernel test failed for 16*16*16*2*2!" << std::endl;
    return;
  }


  if (!r2_test()){
    std::cout << "R2 kernel test failed for 16*16*16*2*2!" << std::endl;
    return;
  }
  */

  std::cout << "All tests passed!" <<std::endl;

  return true;
}
