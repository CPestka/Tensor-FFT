//Functions for the calculation of error properties
#pragma once

#include <math.h>

template<typename Integer, typename float2_t>
double MaxValue(float2_t* data, Integer fft_length){
  double current_largest = 0;
  for(Integer i=0; i<fft_length; i++){
    double tmp = static_cast<double>(data[i].x);
    current_largest = (fabs(tmp) > current_largest) ?
                      fabs(tmp) : current_largest;

    tmp = static_cast<double>(data[i].y);
    current_largest = (fabs(tmp) > current_largest) ?
                      fabs(tmp) : current_largest;
  }

  return current_largest;
}

template<typename Integer, typename float2_t1, typename float2_t2>
double MaxDiv(float2_t1* data_1, float2_t2* data_2, Integer fft_length){
  double current_largest_div = 0;
  for(Integer i=0; i<fft_length; i++){
    double tmp_1 = static_cast<double>(data_1[i].x);
    double tmp_2 = static_cast<double>(data_2[i].x);
    double div = fabs(tmp_1 - tmp_2);
    current_largest_div = (div > current_largest_div) ?
                          div : current_largest_div;

    tmp_1 = static_cast<double>(data_1[i].y);
    tmp_2 = static_cast<double>(data_2[i].y);
    div = fabs(tmp_1 - tmp_2);
    current_largest_div = (div > current_largest_div) ?
                          div : current_largest_div;
  }

  return current_largest_div;
}

template<typename Integer, typename float2_t1, typename float2_t2>
double MeanAbsoluteError(float2_t1* data_1,
                         float2_t2* data_2,
                         Integer fft_length){
  double tmp = 0;

  for(Integer i=0; i<fft_length; i++){
    tmp += fabs(static_cast<double>(data_1[i].x) -
                static_cast<double>(data_2[i].x));
    tmp += fabs(static_cast<double>(data_1[i].y) -
                static_cast<double>(data_2[i].y));
  }

  return tmp / static_cast<double>(2 * fft_length);
}

template<typename Integer, typename float2_t1, typename float2_t2>
double RootMeanSquareError(float2_t1* data_1,
                           float2_t2* data_2,
                           Integer fft_length){
  double tmp = 0;

  for(Integer i=0; i<fft_length; i++){
    double tmp_1 = static_cast<double>(data_1[i].x) -
                   static_cast<double>(data_2[i].x);
    tmp += (tmp_1 * tmp_1);

    tmp_1 = static_cast<double>(data_1[i].y) -
            static_cast<double>(data_2[i].y);
    tmp += (tmp_1 * tmp_1);
  }

  return sqrt(tmp / static_cast<double>(2 * fft_length));
}
