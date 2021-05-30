#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

//Writes results of a fft to file
void WriteResultsToFile(std::string file_name, int fft_length, __half* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float re = data[i];
      float im = data[i + fft_length];
      myfile << i/fft_length << " " << re << " " << im << "\n";
    }
    myfile.close();
  } else {
    std::cout << "Unable to open file " << file_name << std::endl;
  }
}

void WriteBenchResultsToFile(std::vector<double> average,
                             std::vector<double> std_dev,
                             std::string sample_size){
  std::ofstream myfile ("BenchResults" + sample_size + ".dat");
  if (myfile.is_open()) {
    for(int i=0; i<static_cast<int>(average.size()); i++){
      myfile << pow(2, 12 + i) << " "
             << average[i] << " "
             << std_dev[i] << "\n";
    }
    myfile.close();
  } else {
    std::cout << "Unable to open file " << std::endl;
  }
}

void WriteLogToFile(std::string file_name, std::string error_mess){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    myfile << error_mess << "\n";
    myfile.close();
  } else {
    std::cout << "Unable to open file " << file_name << std::endl;
  }
}
