#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

//Writes results of a fft that uses __half to file
void WriteResultsToFile(std::string file_name, int fft_length, __half* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float re = data[i];
      float im = data[i + fft_length];
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << re << " " << im << "\n";
    }
    myfile.close();
  } else {
    std::cout << "Unable to open file " << file_name << std::endl;
  }
}

//Writes results of a fft that uses 2 __half to file
void WriteResultsToFile(std::string file_name, int fft_length, __half* data_RE,
                        __half* data_IM){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float re = data_RE[i];
      float im = data_IM[i];
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << re << " " << im << "\n";
    }
    myfile.close();
  } else {
    std::cout << "Unable to open file " << file_name << std::endl;
  }
}


//Writes results of a fft that uses __half2 to file
void WriteResultsToFileHalf2(std::string file_name, int fft_length,
                             __half2* data){
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    for(int i=0; i<fft_length; i++){
      float x = static_cast<double>(i)/static_cast<double>(fft_length);
      myfile << x << " " << __low2float(data[i]) << " " << __high2float(data[i])
             << "\n";
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
