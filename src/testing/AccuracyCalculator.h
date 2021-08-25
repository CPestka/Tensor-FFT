//These functions read in values from files whose lines have to be formated like
//this: x y z
//The functions than compute the difference of the values in a line between the
//two files and than computes either the average or standard deviation of that
//difference for all lines (for the vector overloads over multiple files).
#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

template<typename Integer>
std::unique_ptr<double[]> ConvertResultsToSplitDouble(Integer fft_length,
    std::unique_ptr<__half[]> data){
  std::unique_ptr<double[]> converted_data =
      std::make_unique<double[]>(2 * fft_length);

  for(Integer i=0; i<(2*fft_length); i++){
    converted_data[i] = static_cast<double>(data[i]);
  }

  return converted_data;
}

//Also scales the results by the fft_length since packed results are obatained
//by cuFFT
template<typename Integer>
std::unique_ptr<double[]> ConvertResultsToSplitDouble(Integer fft_length,
    std::unique_ptr<__half2[]> data){
  std::unique_ptr<double[]> converted_data =
      std::make_unique<double[]>(2 * fft_length);

  for(Integer i=0; i<fft_length; i++){
    converted_data[i] =
        static_cast<double>(data[i].x) / static_cast<double>(fft_length);
    converted_data[i + fft_length] =
        static_cast<double>(data[i].y) / static_cast<double>(fft_length);
  }

  return converted_data;
}

//Also scales the results by the fft_length since packed results are obatained
//by cuFFT
template<typename Integer>
std::unique_ptr<double[]> ConvertResultsToSplitDouble(Integer fft_length,
    std::unique_ptr<cufftComplex[]> data){
  std::unique_ptr<double[]> converted_data =
      std::make_unique<double[]>(2 * fft_length);

  for(Integer i=0; i<fft_length; i++){
    converted_data[i] =
        static_cast<double>(data[i].x) / static_cast<double>(fft_length);
    converted_data[i + fft_length] =
        static_cast<double>(data[i].y) / static_cast<double>(fft_length);
  }

  return converted_data;
}

//Also scales the results by the fft_length since packed results are obatained
//by cuFFT
template<typename Integer>
std::unique_ptr<double[]> ConvertResultsToSplitDouble(Integer fft_length,
    std::unique_ptr<cufftDoubleComplex[]> data){
  std::unique_ptr<double[]> converted_data =
      std::make_unique<double[]>(2 * fft_length);

  for(Integer i=0; i<fft_length; i++){
    converted_data[i] =
        static_cast<double>(data[i].x) / static_cast<double>(fft_length);
    converted_data[i + fft_length] =
        static_cast<double>(data[i].y) / static_cast<double>(fft_length);
  }

  return converted_data;
}

template<typename Integer>
double GetLargestDeviation(const double* data_1, const double* data_2,
                           Integer fft_length){
  double largest_dev = 0;

  for(Integer i=0; i<fft_length; i++){
    double new_dev = fabs(data_1[i] - data_2[i]);
    largest_dev = new_dev > largest_dev ? new_dev : largest_dev;

    new_dev = fabs(data_1[i + fft_length] - data_2[i + fft_length]);
    largest_dev = new_dev > largest_dev ? new_dev : largest_dev;
  }

  return largest_dev;
}


template<typename Integer>
double ComputeAverageDeviation(const double* data_1, const double* data_2,
                               const Integer fft_length){
  std::unique_ptr<double[]> deviations =
      std::make_unique<double[]>(fft_length * 2);

  for(Integer i=0; i<(2 * fft_length); i++){
    deviations[i] = fabs(data_1[i] - data_2[i]);
  }

  //In place cascade summation of deviations
  Integer tmp = fft_length;
  while (tmp != 1) {
    for(Integer i=0; i<tmp; i++){
      deviations[i] = deviations[i] + deviations[i + tmp];
    }
    //FFT length is a power of 2
    tmp = tmp / 2;
  }

  return (deviations[0] / static_cast<double>(2 * fft_length));
}

template<typename Integer>
double ComputeSigmaOfDeviation(const double* data_1, const double* data_2,
                               const Integer fft_length, const double average){
  std::unique_ptr<double[]> deviations =
      std::make_unique<double[]>(fft_length * 2);

  for(Integer i=0; i<(2 * fft_length); i++){
    double tmp_1 = fabs(data_1[i] - data_2[i]) - average;
    deviations[i] = tmp_1 * tmp_1;
  }

  //In place cascade summation of deviations
  Integer tmp = fft_length;
  while (tmp != 1) {
    for(Integer i=0; i<tmp; i++){
      deviations[i] = deviations[i] + deviations[i + tmp];
    }
    //FFT length is a power of 2
    tmp = tmp / 2;
  }

  return sqrt(deviations[0] / static_cast<double>((2 * fft_length) - 1));
}

double GetLargestDeviation(const std::string file_name_1,
                           const std::string file_name_2){
  double largest_dev = 0;

  std::string line_1;
  std::string line_2;
  std::ifstream file_1(file_name_1);
  std::ifstream file_2(file_name_2);

  if (file_1.is_open() && file_2.is_open()){
    while (std::getline(file_1, line_1)) {
      if (!std::getline(file_2, line_2)) {
        std::cout << "Error failed to read line!" << std::endl;
      }

      std::stringstream ss1;
      std::stringstream ss2;
      std::string tmp_string;

      double tmp1_RE = 0;
      double tmp1_IM = 0;
      double tmp2_RE = 0;
      double tmp2_IM = 0;

      ss1 << line_1;

      ss1 >> tmp_string;
      tmp_string = "";
      ss1 >> tmp_string;
      tmp1_RE = static_cast<double>(std::stof(tmp_string));
      tmp_string = "";
      ss1 >> tmp_string;
      tmp1_IM = static_cast<double>(std::stof(tmp_string));

      ss2 << line_2;

      ss2 >> tmp_string;
      tmp_string = "";
      ss2 >> tmp_string;
      tmp2_RE = static_cast<double>(std::stof(tmp_string));
      tmp_string = "";
      ss2 >> tmp_string;
      tmp2_IM = static_cast<double>(std::stof(tmp_string));

      double new_dev = fabs(tmp1_RE - tmp2_RE);
      largest_dev = (new_dev > largest_dev) ? new_dev : largest_dev;
      new_dev = fabs(tmp1_IM - tmp2_IM);
      largest_dev = (new_dev > largest_dev) ? new_dev : largest_dev;
    }
  }
  return largest_dev;
}

double GetLargestDeviation(const std::vector<std::string> file_name_1,
                           const std::vector<std::string> file_name_2){
  double largest_dev = 0;

  std::string line_1;
  std::string line_2;

  for(int i=0; i<static_cast<int>(file_name_1.size()); i++){
    std::ifstream file_1(file_name_1[i]);
    std::ifstream file_2(file_name_2[i]);

    if (file_1.is_open() && file_2.is_open()){
      while (std::getline(file_1, line_1)) {
        if (!std::getline(file_2, line_2)) {
          std::cout << "Error failed to read line!" << std::endl;
        }

        std::stringstream ss1;
        std::stringstream ss2;
        std::string tmp_string;

        double tmp1_RE = 0;
        double tmp1_IM = 0;
        double tmp2_RE = 0;
        double tmp2_IM = 0;

        ss1 << line_1;

        ss1 >> tmp_string;
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_IM = static_cast<double>(std::stof(tmp_string));

        ss2 << line_2;

        ss2 >> tmp_string;
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_IM = static_cast<double>(std::stof(tmp_string));

        double new_dev = fabs(tmp1_RE - tmp2_RE);
        largest_dev = (new_dev > largest_dev) ? new_dev : largest_dev;
        new_dev = fabs(tmp1_IM - tmp2_IM);
        largest_dev = (new_dev > largest_dev) ? new_dev : largest_dev;
      }
    }
  }

  return largest_dev;
}

double GetLargestDeviation(const std::vector<std::string> file_name_1,
                           const std::string file_name_2){
  double largest_dev = 0;

  std::string line_1;
  std::string line_2;

  for(int i=0; i<static_cast<int>(file_name_1.size()); i++){
    std::ifstream file_1(file_name_1[i]);
    std::ifstream file_2(file_name_2);

    if (file_1.is_open() && file_2.is_open()){
      while (std::getline(file_1, line_1)) {
        if (!std::getline(file_2, line_2)) {
          std::cout << "Error failed to read line!" << std::endl;
        }

        std::stringstream ss1;
        std::stringstream ss2;
        std::string tmp_string;

        double tmp1_RE = 0;
        double tmp1_IM = 0;
        double tmp2_RE = 0;
        double tmp2_IM = 0;

        ss1 << line_1;

        ss1 >> tmp_string;
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_IM = static_cast<double>(std::stof(tmp_string));

        ss2 << line_2;

        ss2 >> tmp_string;
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_IM = static_cast<double>(std::stof(tmp_string));

        double new_dev = fabs(tmp1_RE - tmp2_RE);
        largest_dev = (new_dev > largest_dev) ? new_dev : largest_dev;
        new_dev = fabs(tmp1_IM - tmp2_IM);
        largest_dev = (new_dev > largest_dev) ? new_dev : largest_dev;
      }
    }
  }

  return largest_dev;
}

double ComputeAverageDeviation(const std::string file_name_1,
                               const std::string file_name_2){
  long long length = 0;
  double tmp_dev = 0;

  std::string line_1;
  std::string line_2;
  std::ifstream file_1(file_name_1);
  std::ifstream file_2(file_name_2);

  if (file_1.is_open() && file_2.is_open()){
    while (std::getline(file_1, line_1)) {
      if (!std::getline(file_2, line_2)) {
        std::cout << "Error failed to read line!" << std::endl;
      }
      length ++;

      std::stringstream ss1;
      std::stringstream ss2;
      std::string tmp_string;

      double tmp1_RE = 0;
      double tmp1_IM = 0;
      double tmp2_RE = 0;
      double tmp2_IM = 0;

      ss1 << line_1;

      ss1 >> tmp_string;
      tmp_string = "";
      ss1 >> tmp_string;
      tmp1_RE = static_cast<double>(std::stof(tmp_string));
      tmp_string = "";
      ss1 >> tmp_string;
      tmp1_IM = static_cast<double>(std::stof(tmp_string));

      ss2 << line_2;

      ss2 >> tmp_string;
      tmp_string = "";
      ss2 >> tmp_string;
      tmp2_RE = static_cast<double>(std::stof(tmp_string));
      tmp_string = "";
      ss2 >> tmp_string;
      tmp2_IM = static_cast<double>(std::stof(tmp_string));

      tmp_dev += fabs(tmp1_RE - tmp2_RE);
      tmp_dev += fabs(tmp1_IM - tmp2_IM);
    }
  }
  return (tmp_dev / static_cast<double>(2 * length));
}

double ComputeSigmaOfDeviation(const std::string file_name_1,
                               const std::string file_name_2,
                               const double average){
  long long length = 0;
  double tmp_dev = 0;

  std::string line_1;
  std::string line_2;
  std::ifstream file_1(file_name_1);
  std::ifstream file_2(file_name_2);

  if (file_1.is_open() && file_2.is_open()){
    while (std::getline(file_1, line_1)) {
      if (!std::getline(file_2, line_2)) {
        std::cout << "Error failed to read line!" << std::endl;
      }
      length ++;

      std::stringstream ss1;
      std::stringstream ss2;
      std::string tmp_string;

      double tmp1_RE = 0;
      double tmp1_IM = 0;
      double tmp2_RE = 0;
      double tmp2_IM = 0;

      ss1 << line_1;

      ss1 >> tmp_string;
      tmp_string = "";
      ss1 >> tmp_string;
      tmp1_RE = static_cast<double>(std::stof(tmp_string));
      tmp_string = "";
      ss1 >> tmp_string;
      tmp1_IM = static_cast<double>(std::stof(tmp_string));

      ss2 << line_2;

      ss2 >> tmp_string;
      tmp_string = "";
      ss2 >> tmp_string;
      tmp2_RE = static_cast<double>(std::stof(tmp_string));
      tmp_string = "";
      ss2 >> tmp_string;
      tmp2_IM = static_cast<double>(std::stof(tmp_string));

      double tmp = fabs(fabs(tmp1_RE - tmp2_RE) - average);
      tmp_dev += (tmp * tmp);
      tmp = fabs(fabs(tmp1_IM - tmp2_IM) - average);
      tmp_dev += (tmp * tmp);
    }
  }
  return sqrt(tmp_dev / static_cast<double>((2 * length) - 1));
}

double ComputeAverageDeviation(const std::vector<std::string> file_name_1,
                               const std::vector<std::string> file_name_2){
  long long length = 0;
  double tmp_dev = 0;

  std::string line_1;
  std::string line_2;

  for(int i=0; i<static_cast<int>(file_name_1.size()); i++){
    std::ifstream file_1(file_name_1[i]);
    std::ifstream file_2(file_name_2[i]);

    if (file_1.is_open() && file_2.is_open()){
      while (std::getline(file_1, line_1)) {
        if (!std::getline(file_2, line_2)) {
          std::cout << "Error failed to read line!" << std::endl;
        }
        length ++;

        std::stringstream ss1;
        std::stringstream ss2;
        std::string tmp_string;

        double tmp1_RE = 0;
        double tmp1_IM = 0;
        double tmp2_RE = 0;
        double tmp2_IM = 0;

        ss1 << line_1;

        ss1 >> tmp_string;
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_IM = static_cast<double>(std::stof(tmp_string));

        ss2 << line_2;

        ss2 >> tmp_string;
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_IM = static_cast<double>(std::stof(tmp_string));

        tmp_dev += fabs(tmp1_RE - tmp2_RE);
        tmp_dev += fabs(tmp1_IM - tmp2_IM);
      }
    }
  }

  return (tmp_dev / static_cast<double>(2 * length));
}

double ComputeSigmaOfDeviation(const std::vector<std::string> file_name_1,
                               const std::vector<std::string> file_name_2,
                               const double average){
  long long length = 0;
  double tmp_dev = 0;

  std::string line_1;
  std::string line_2;

  for(int i=0; i<static_cast<int>(file_name_1.size()); i++){
    std::ifstream file_1(file_name_1[i]);
    std::ifstream file_2(file_name_2[i]);

    if (file_1.is_open() && file_2.is_open()){
      while (std::getline(file_1, line_1)) {
        if (!std::getline(file_2, line_2)) {
          std::cout << "Error failed to read line!" << std::endl;
        }
        length ++;

        std::stringstream ss1;
        std::stringstream ss2;
        std::string tmp_string;

        double tmp1_RE = 0;
        double tmp1_IM = 0;
        double tmp2_RE = 0;
        double tmp2_IM = 0;

        ss1 << line_1;

        ss1 >> tmp_string;
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_IM = static_cast<double>(std::stof(tmp_string));

        ss2 << line_2;

        ss2 >> tmp_string;
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_IM = static_cast<double>(std::stof(tmp_string));

        double tmp = fabs(fabs(tmp1_RE - tmp2_RE) - average);
        tmp_dev += (tmp * tmp);
        tmp = fabs(fabs(tmp1_IM - tmp2_IM) - average);
        tmp_dev += (tmp * tmp);
      }
    }
  }

  return sqrt(tmp_dev / static_cast<double>((2 * length) - 1));
}

double ComputeAverageDeviation(const std::vector<std::string> file_name_1,
                               const std::string file_name_2){
  long long length = 0;
  double tmp_dev = 0;

  std::string line_1;
  std::string line_2;

  for(int i=0; i<static_cast<int>(file_name_1.size()); i++){
    std::ifstream file_1(file_name_1[i]);
    std::ifstream file_2(file_name_2);

    if (file_1.is_open() && file_2.is_open()){
      while (std::getline(file_1, line_1)) {
        if (!std::getline(file_2, line_2)) {
          std::cout << "Error failed to read line!" << std::endl;
        }
        length ++;

        std::stringstream ss1;
        std::stringstream ss2;
        std::string tmp_string;

        double tmp1_RE = 0;
        double tmp1_IM = 0;
        double tmp2_RE = 0;
        double tmp2_IM = 0;

        ss1 << line_1;

        ss1 >> tmp_string;
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_IM = static_cast<double>(std::stof(tmp_string));

        ss2 << line_2;

        ss2 >> tmp_string;
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_IM = static_cast<double>(std::stof(tmp_string));

        tmp_dev += fabs(tmp1_RE - tmp2_RE);
        tmp_dev += fabs(tmp1_IM - tmp2_IM);
      }
    }
  }

  return (tmp_dev / static_cast<double>(2 * length));
}

double ComputeSigmaOfDeviation(const std::vector<std::string> file_name_1,
                               const std::string file_name_2,
                               const double average){
  long long length = 0;
  double tmp_dev = 0;

  std::string line_1;
  std::string line_2;

  for(int i=0; i<static_cast<int>(file_name_1.size()); i++){
    std::ifstream file_1(file_name_1[i]);
    std::ifstream file_2(file_name_2);

    if (file_1.is_open() && file_2.is_open()){
      while (std::getline(file_1, line_1)) {
        if (!std::getline(file_2, line_2)) {
          std::cout << "Error failed to read line!" << std::endl;
        }
        length ++;

        std::stringstream ss1;
        std::stringstream ss2;
        std::string tmp_string;

        double tmp1_RE = 0;
        double tmp1_IM = 0;
        double tmp2_RE = 0;
        double tmp2_IM = 0;

        ss1 << line_1;

        ss1 >> tmp_string;
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss1 >> tmp_string;
        tmp1_IM = static_cast<double>(std::stof(tmp_string));

        ss2 << line_2;

        ss2 >> tmp_string;
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_RE = static_cast<double>(std::stof(tmp_string));
        tmp_string = "";
        ss2 >> tmp_string;
        tmp2_IM = static_cast<double>(std::stof(tmp_string));

        double tmp = fabs(fabs(tmp1_RE - tmp2_RE) - average);
        tmp_dev += (tmp * tmp);
        tmp = fabs(fabs(tmp1_IM - tmp2_IM) - average);
        tmp_dev += (tmp * tmp);
      }
    }
  }

  return sqrt(tmp_dev / static_cast<double>((2 * length) - 1));
}
