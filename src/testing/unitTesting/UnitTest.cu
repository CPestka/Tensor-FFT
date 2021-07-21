//Used to test functonality
#include <iostream>
#include <string>

#include "TransposeTest.cu"
#include "DFTMatrixTest.cu"
#include "DFTTest.cu"
#include "FFTTest.cu"

int main(){
  if (!TestTranspose16()){
    std::cout << "Transpose 16*16*16 test failed!" << std::endl;
    return false;
  }
  if (!TestTranspose16_2()){
    std::cout << "Transpose 16*16*16*2*2 test failed!" << std::endl;
    return false;
  }

  if (!TestDFTMatrix()){
    std::cout << "DFT matrix test failed!" << std::endl;
    return false;
  }
  if (!TestDFTMatrixBatch()){
    std::cout << "DFT matrix test 16*16*2 failed!" << std::endl;
    return false;
  }

  if (!TestDFTKernel_0()){
    std::cout << "DFT kernel test failed for 16*16*16 with value 0!"
              << std::endl;
    return false;
  }
  if (!TestDFTKernelSin_16()){
    std::cout << "DFT kernel test failed for 16*16*16 with sin(x) x [0:2*PI]!"
              << std::endl;
    return false;
  }
  if (!TestDFTKernelSin_2()){
    std::cout << "DFT kernel test failed for 16*16*16*16*2 with sin(x) x"
              << " [0:2*PI]!"
              << std::endl;
    return false;
  }

  if (!TestFullFFT(16*16, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16*16 failed." << std::endl;
    return false;
  }
  if (!TestFullFFT(16*16*16, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16*16*16 failed." << std::endl;
    return false;
  }
  if (!TestFullFFT(16*16*16*16*16, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16^5 failed." << std::endl;
    return false;
  }
  if (!TestFullFFT(16*16*16*2*2*2, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16^3*2^3 failed." << std::endl;
    return false;
  }
  if (!TestFullFFT(16*16*16*16*16*2*2*2, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16^5*2^3 failed." << std::endl;
    return false;
  }

  if (!TestFullFFTAsynch(16*16, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16*16 failed." << std::endl;
    return false;
  }
  if (!TestFullFFTAsynch(16*16*16, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16*16*16 failed." << std::endl;
    return false;
  }
  if (!TestFullFFTAsynch(16*16*16*16*16, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16^5 failed." << std::endl;
    return false;
  }
  if (!TestFullFFTAsynch(16*16*16*2*2*2, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16^3*2^3 failed." << std::endl;
    return false;
  }
  if (!TestFullFFTAsynch(16*16*16*16*16*2*2*2, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16^5*2^3 failed." << std::endl;
    return false;
  }


  std::cout << "All tests passed!" <<std::endl;

  return true;
}
