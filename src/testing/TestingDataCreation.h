//Contains functions that produce example input data for testing of the ffts
#pragma once

#include <memory>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>

std::vector<float> GetRandomWeights(int max_frequencies, int seed){
  std::seed_seq seed_seq = {seed};
  std::default_random_engine generator(seed_seq);
  std::uniform_real_distribution<float> distro(-1.0, 1.0);

  std::vector<float> results;

  for(int i=0; i<max_frequencies; i++){
    results.push_back(distro(generator));
  }

  return results;
}



//Creates half precission data which is a superpostion of the a_i * sin(2^i * x)
//with x [0:1] and a_i provided by weights
//Has 2*amount_of_timesamples elements, with the RE ones in the first half
std::unique_ptr<__half[]> CreateSineSuperpostion(int amount_of_timesamples,
                                                 std::vector<float> weights){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(
      2 * amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights.size()); j++){
      tmp += (weights[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i] = __float2half(tmp);
    data[i + amount_of_timesamples] = 0;
  }
  return data;
}

//Creates half precission data which is a superpostion of the a_i * sin(2^i * x)
//with x [0:1] and a_i provided by weights
//Has 2*amount_of_timesamples elements, with the RE ones in the first half
std::unique_ptr<__half[]> CreateSineSuperpostion(int amount_of_timesamples,
                                                 std::vector<float> weights_RE,
                                                 std::vector<float> weights_IM){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(
      2 * amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights_RE.size()); j++){
      tmp += (weights_RE[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i] = __float2half(tmp);

    tmp = 0;
    for(int j=0; j<static_cast<int>(weights_IM.size()); j++){
      tmp += (weights_IM[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i + amount_of_timesamples] = __float2half(tmp);
  }
  return data;
}

//Returns ptr to zero valued data
std::unique_ptr<__half[]> CreateZeros(int length){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(2 * length);

  for(int i=0; i<length; i++){
    data[i] = 0;
    data[i + length] = 0;
  }
  return data;
}

//Creates half precission data which is a superpostion of the a_i * sin(2^i * x)
//with x [0:1] and a_i provided by weights
//Has amount_of_timesamples half2 elements holding one complex value each
std::unique_ptr<__half2[]> CreateSineSuperpostionH2(int amount_of_timesamples,
                                                    std::vector<float> weights){
  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
      amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights.size()); j++){
      tmp += (weights[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i] = __floats2half2_rn(tmp, 0);
  }
  return data;
}

//Creates half precission data which is a superpostion of the a_i * sin(2^i * x)
//with x [0:1] and a_i provided by weights
//Has amount_of_timesamples half2 elements holding one complex value each
std::unique_ptr<__half2[]> CreateSineSuperpostionH2(
    int amount_of_timesamples,
    std::vector<float> weights_RE,
    std::vector<float> weights_IM){
  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
      amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights_RE.size()); j++){
      tmp += (weights_RE[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }

    float tmp_1 = 0;
    for(int j=0; j<static_cast<int>(weights_IM.size()); j++){
      tmp_1 += (weights_IM[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }

    data[i] = __floats2half2_rn(tmp, tmp_1);
  }
  return data;
}

//Like CreateSineSuperpostionH2, but produces batch_size times the same data
//sequential in memory
std::unique_ptr<__half2[]> CreateSineSuperpostionH2Batch(
    int amount_of_timesamples, std::vector<float> weights, int batch_size){

  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
      amount_of_timesamples * batch_size);

  for(int k=0; k<batch_size; k++){
    for(int i=0; i<amount_of_timesamples; i++){
      float tmp = 0;
      for(int j=0; j<static_cast<int>(weights.size()); j++){
        tmp += (weights[j] *
                sin(2 * M_PI * std::pow(2, j) *
                    (static_cast<double>(i) / amount_of_timesamples)));
      }
      data[i + (k * amount_of_timesamples)] = __floats2half2_rn(tmp, 0);
    }
  }

  return data;
}

//Like CreateSineSuperpostionH2, but produces batch_size times the same data
//sequential in memory
std::unique_ptr<__half2[]> CreateSineSuperpostionH2Batch(
    int amount_of_timesamples, std::vector<float> weights_RE,
    std::vector<float> weights_IM, int batch_size){

  std::unique_ptr<__half2[]> data = std::make_unique<__half2[]>(
      amount_of_timesamples * batch_size);

  for(int k=0; k<batch_size; k++){
    for(int i=0; i<amount_of_timesamples; i++){
      float tmp = 0;
      for(int j=0; j<static_cast<int>(weights_RE.size()); j++){
        tmp += (weights_RE[j] *
                sin(2 * M_PI * std::pow(2, j) *
                    (static_cast<double>(i) / amount_of_timesamples)));
      }
      float tmp_1 = 0;
      for(int j=0; j<static_cast<int>(weights_IM.size()); j++){
        tmp_1 += (weights_IM[j] *
                  sin(2 * M_PI * std::pow(2, j) *
                      (static_cast<double>(i) / amount_of_timesamples)));
      }
      data[i + (k * amount_of_timesamples)] = __floats2half2_rn(tmp, tmp_1);
    }
  }

  return data;
}

//Creates double precission data which is a superpostion of the a_i *
//sin(2^i * x) with x [0:1] and a_i provided by weights
//Has amount_of_timesamples cufftComplex elements holding one complex value each
std::unique_ptr<cufftDoubleComplex[]> CreateSineSuperpostionDouble(
    int amount_of_timesamples, std::vector<float> weights){
  std::unique_ptr<cufftDoubleComplex[]> data =
      std::make_unique<cufftDoubleComplex[]>(amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights.size()); j++){
      tmp += (weights[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i].x = tmp;
    data[i].y = 0;
  }
  return data;
}

//Creates double precission data which is a superpostion of the a_i *
//sin(2^i * x) with x [0:1] and a_i provided by weights
//Has amount_of_timesamples cufftComplex elements holding one complex value each
std::unique_ptr<cufftDoubleComplex[]> CreateSineSuperpostionDouble(
    int amount_of_timesamples, std::vector<float> weights_RE,
    std::vector<float> weights_IM){
  std::unique_ptr<cufftDoubleComplex[]> data =
      std::make_unique<cufftDoubleComplex[]>(amount_of_timesamples);

  for(int i=0; i<amount_of_timesamples; i++){
    float tmp = 0;
    for(int j=0; j<static_cast<int>(weights_RE.size()); j++){
      tmp += (weights_RE[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i].x = tmp;

    tmp = 0;
    for(int j=0; j<static_cast<int>(weights_IM.size()); j++){
      tmp += (weights_IM[j] *
              sin(2 * M_PI * std::pow(2, j) *
                  (static_cast<double>(i) / amount_of_timesamples)));
    }
    data[i].y = tmp;
  }
  return data;
}

std::unique_ptr<__half[]> CreateSineSuperpostionBatch(
    int amount_of_timesamples, int amount_of_batches,
    std::vector<float> weights){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(
      2 * amount_of_timesamples * amount_of_batches);

  for(int k=0; k<amount_of_batches; k++){
    for(int i=0; i<amount_of_timesamples; i++){
      float tmp = 0;
      for(int j=0; j<static_cast<int>(weights.size()); j++){
        tmp += (weights[j] *
                sin(2 * M_PI * std::pow(2, j) *
                    (static_cast<double>(i) / amount_of_timesamples)));
      }
      data[i + (k * 2 * amount_of_timesamples)] = __float2half(tmp);
      data[i + amount_of_timesamples + (k * 2 * amount_of_timesamples)] = 0;
    }
  }

  return data;
}

std::unique_ptr<__half[]> CreateSineSuperpostionBatch(
    int amount_of_timesamples, int amount_of_batches,
    std::vector<float> weights_RE, std::vector<float> weights_IM){
  std::unique_ptr<__half[]> data = std::make_unique<__half[]>(
      2 * amount_of_timesamples * amount_of_batches);

  for(int k=0; k<amount_of_batches; k++){
    for(int i=0; i<amount_of_timesamples; i++){
      float tmp = 0;
      for(int j=0; j<static_cast<int>(weights_RE.size()); j++){
        tmp += (weights_RE[j] *
                sin(2 * M_PI * std::pow(2, j) *
                    (static_cast<double>(i) / amount_of_timesamples)));
      }
      data[i + (k * 2 * amount_of_timesamples)] = __float2half(tmp);

      tmp = 0;
      for(int j=0; j<static_cast<int>(weights_IM.size()); j++){
        tmp += (weights_IM[j] *
                sin(2 * M_PI * std::pow(2, j) *
                    (static_cast<double>(i) / amount_of_timesamples)));
      }
      data[i + amount_of_timesamples + (k * 2 * amount_of_timesamples)] =
          __float2half(tmp);
    }
  }

  return data;
}
