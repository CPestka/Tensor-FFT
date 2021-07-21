#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

double ComputeAverageDeviation(std::string file_name_1,
                               std::string file_name_2){
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
      tmp = "";
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

double ComputeSigmaOfDeviation(std::string file_name_1,
                               std::string file_name_2,
                               double average){
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

double ComputeAverageDeviation(std::vector<std::string> file_name_1,
                               std::vector<std::string> file_name_2){
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

double ComputeSigmaOfDeviation(std::vector<std::string> file_name_1,
                               std::vector<std::string> file_name_2,
                               double average){
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
        tmp = fabs(fabs(tmp1_IM - tmp2_IM) - average)
        tmp_dev += (tmp * tmp);
      }
    }
  }

  return sqrt(tmp_dev / static_cast<double>((2 * length) - 1));
}

double ComputeAverageDeviation(std::vector<std::string> file_name_1,
                               std::string file_name_2){
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

double ComputeSigmaOfDeviation(std::vector<std::string> file_name_1,
                               std::string file_name_2,
                               double average){
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
        tmp = fabs(fabs(tmp1_IM - tmp2_IM) - average)
        tmp_dev += (tmp * tmp);
      }
    }
  }

  return sqrt(tmp_dev / static_cast<double>((2 * length) - 1));
}
