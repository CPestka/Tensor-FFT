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
  } else {
    std::cout << "Transpose 16*16*16 test completed successfully!" << std::endl;
  }
  if (!TestTranspose16_2()){
    std::cout << "Transpose 16*16*16*2*2 test failed!" << std::endl;
    return false;
  } else {
    std::cout << "Transpose 16*16*16*2*2 test completed successfully!"
              << std::endl;
  }

  if (!TestDFTMatrix()){
    std::cout << "DFT matrix test failed!" << std::endl;
    return false;
  } else {
    std::cout << "DFT matrix test completed successfully!" << std::endl;
  }
  if (!TestDFTMatrixBatch()){
    std::cout << "DFT matrix test 16*16*2 failed!" << std::endl;
    return false;
  } else {
    std::cout << "DFT matrix test 16*16*2 completed successfully!" << std::endl;
  }

  if (!TestDFTKernel_0()){
    std::cout << "DFT kernel test failed for 16*16*16 with value 0!"
              << std::endl;
    return false;
  } else {
    std::cout << "DFT kernel test for 16*16*16 with value 0 completed"
              << " successfully!" << std::endl;
  }
  if (!TestDFTKernelSin_16()){
    std::cout << "DFT kernel test failed for 16*16*16 with sin(x) x [0:2*PI]!"
              << std::endl;
    return false;
  } else {
    std::cout << "DFT kernel test for 16*16*16 with sin(x) x [0:2*PI] completed"
              << " successfully!" << std::endl;
  }
  if (!TestDFTKernelSin_2()){
    std::cout << "DFT kernel test failed for 16*16*16*16*2 with sin(x) x"
              << " [0:2*PI]!"
              << std::endl;
    return false;
  } else {
    std::cout << "DFT kernel test for 16*16*16*16*2 with sin(x) x"
              << " [0:2*PI] completed successfully!"
              << std::endl;
  }

  //Currently not active since comparission data cant be generated via cuFFT
  //since the length is to short for it
  if (!TestFullFFT(16*16, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16*16 failed." << std::endl;
    return false;
  } else {
    std::cout << "FFT test for a length of 16*16 completed successfully."
              << std::endl;
  }
  if (!TestFullFFT(16*16*16, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16*16*16 failed." << std::endl;
    return false;
  } else {
    std::cout << "FFT test for a length of 16*16*16 completed successfully."
              << std::endl;
  }
  if (!TestFullFFT(16*16*16*16*16, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16^5 failed." << std::endl;
    return false;
  } else {
    std::cout << "FFT test for a length of 16^5 completed successfully."
              << std::endl;
  }
  if (!TestFullFFT(16*16*16*2*2*2, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16^3*2^3 failed." << std::endl;
    return false;
  } else {
    std::cout << "FFT test for a length of 16^3*2^3 completed successfully."
              << std::endl;
  }
  if (!TestFullFFT(16*16*16*16*16*2*2*2, 0.1, 0.1)){
    std::cout << "FFT test for a length of 16^5*2^3 failed." << std::endl;
    return false;
  } else {
    std::cout << "FFT test for a length of 16^5*2^3 completed successfully."
              << std::endl;
  }

  if (!TestFullFFTAsynch(16*16, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16*16 failed." << std::endl;
    return false;
  } else {
    std::cout << "Async FFT test for a length of 16*16 completed successfully."
              << std::endl;
  }
  if (!TestFullFFTAsynch(16*16*16, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16*16*16 failed." << std::endl;
    return false;
  } else {
    std::cout << "Async FFT test for a length of 16*16*16 completed"
              << " successfully." << std::endl;
  }
  if (!TestFullFFTAsynch(16*16*16*16*16, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16^5 failed." << std::endl;
    return false;
  } else {
    std::cout << "Async FFT test for a length of 16^5 completed successfully."
              << std::endl;
  }
  if (!TestFullFFTAsynch(16*16*16*2*2*2, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16^3*2^3 failed." << std::endl;
    return false;
  } else {
    std::cout << "Async FFT test for a length of 16^3*2^3 completed"
              << " successfully." << std::endl;
  }
  if (!TestFullFFTAsynch(16*16*16*16*16*2*2*2, 4, 0.1, 0.1)){
    std::cout << "Async FFT test for a length of 16^5*2^3 failed." << std::endl;
    return false;
  } else {
    std::cout << "Async FFT test for a length of 16^5*2^3 completed"
              << " successfully." << std::endl;
  }

  std::cout << "All tests passed!" <<std::endl;

  return true;
}
