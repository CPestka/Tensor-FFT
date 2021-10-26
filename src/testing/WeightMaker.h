//Contains functions that return different sets of weights for the purpose of
//creating example data for the FFT
#pragma once

#include <random>

void SetRandomWeights(float2* weights, int max_frequencies, int seed){
  std::seed_seq seed_seq = {seed};
  std::default_random_engine generator(seed_seq);
  std::uniform_real_distribution<float> distro(-1.0, 1.0);

  for(int i=0; i<max_frequencies; i++){
    weights[i].x = distro(generator);
    weights[i].y = distro(generator);
  }
}

void SetRandomWeightsRE(float2* weights, int max_frequencies, int seed){
  std::seed_seq seed_seq = {seed};
  std::default_random_engine generator(seed_seq);
  std::uniform_real_distribution<float> distro(-1.0, 1.0);

  for(int i=0; i<max_frequencies; i++){
    weights[i].x = distro(generator);
    weights[i].y = 0;
  }
}

void SetRandomWeightsIM(float2* weights, int max_frequencies, int seed){
  std::seed_seq seed_seq = {seed};
  std::default_random_engine generator(seed_seq);
  std::uniform_real_distribution<float> distro(-1.0, 1.0);

  for(int i=0; i<max_frequencies; i++){
    weights[i].x = 0;
    weights[i].y = distro(generator);
  }
}

//For testing

void SetDummyWeights(float2* weights){
  weights[0].x = 1.0;
  weights[0].y = 0.2;
  weights[1].x = 0.5;
  weights[1].y = 0.7;
  weights[2].x = 0.1;
  weights[2].y = 0.1;
  weights[3].x = 1.0;
  weights[3].y = 0.9;
  weights[4].x = 0.7;
  weights[4].y = 0.1;
}

void SetDummyWeightsRE(float2* weights){
  weights[0].x = 1.0;
  weights[0].y = 0.0;
  weights[1].x = 0.5;
  weights[1].y = 0.0;
  weights[2].x = 0.1;
  weights[2].y = 0.0;
  weights[3].x = 1.0;
  weights[3].y = 0.0;
  weights[4].x = 0.7;
  weights[4].y = 0.0;
}

void SetDummyWeightsRE1(float2* weights){
  weights[0].x = 1.0;
  weights[0].y = 0.0;
  weights[1].x = 1.0;
  weights[1].y = 0.0;
  weights[2].x = 1.0;
  weights[2].y = 0.0;
  weights[3].x = 1.0;
  weights[3].y = 0.0;
  weights[4].x = 1.0;
  weights[4].y = 0.0;
}

void SetDummyWeightsIM(float2* weights){
  weights[0].x = 0.0;
  weights[0].y = 0.2;
  weights[1].x = 0.0;
  weights[1].y = 0.7;
  weights[2].x = 0.0;
  weights[2].y = 0.1;
  weights[3].x = 0.0;
  weights[3].y = 0.9;
  weights[4].x = 0.0;
  weights[4].y = 0.1;
}
