//Used to test correctness of transposer
#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "TestingDataCreation.cu"
#include "FileWriter.cu"
#include "../base/Transposer.cu"

bool transpose16_test(){
  int fft_length = 16*16*16;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data_1 =
      CreateSineSuperpostion(fft_length, weights);
  std::unique_ptr<__half[]> data_2 =
      CreateSineSuperpostion(fft_length, weights);

  WriteResultsToFile("input.dat", fft_length, data_1.get());

  __half old1_RE[16*16][16];
  __half old1_IM[16*16][16];
  __half new1_RE[16][16*16];
  __half new1_IM[16][16*16];

  for(int i=0; i<16*16; i++){
    for(int j=0; j<16; j++){
      old1_RE[i][j] = data_2[j + 16 * i];
      old1_IM[i][j] = data_2[j + 16 * i + fft_length];
    }
  }

  for(int i=0; i<16; i++){
    for(int j=0; j<16*16; j++){
      new1_RE[i][j] = old1_RE[j][i];
      new1_IM[i][j] = old1_IM[j][i];
    }
  }

  __half old2_RE[16][16][16];
  __half old2_IM[16][16][16];
  __half new2_RE[16][16][16];
  __half new2_IM[16][16][16];

  for(int i=0; i<16; i++){
    for(int j=0; j<16; j++){
      for(int k=0; k<16; k++){
        old2_RE[i][j][k] = new1_RE[i][k + 16 * j];
        old2_IM[i][j][k] = new1_IM[i][k + 16 * j];
      }
    }
  }

  for(int i=0; i<16; i++){
    for(int j=0; j<16; j++){
      for(int k=0; k<16; k++){
        new2_RE[i][j][k] = old2_RE[i][k][j];
        new2_IM[i][j][k] = old2_IM[i][k][j];
      }
    }
  }

  for(int i=0; i<16; i++){
    for(int j=0; j<16; j++){
      for(int k=0; k<16; k++){
        data_2[k + 16*j + 16*16*i] = new2_RE[i][j][k];
        data_2[k + 16*j + 16*16*i + fft_length] = new2_IM[i][j][k];
      }
    }
  }

  WriteResultsToFile("transposed_test_cpu.dat", fft_length, data_2.get());

  __half* dptr_input_RE;
  __half* dptr_input_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  cudaMalloc((void**)(&dptr_input_RE), 4 * sizeof(__half) * fft_length);

  dptr_input_IM = dptr_input_RE + fft_length;
  dptr_results_RE = dptr_input_IM + fft_length;
  dptr_results_IM = dptr_results_RE + fft_length;

  cudaMemcpy(dptr_input_RE, data_1.get(), 2 * fft_length * sizeof(__half),
             cudaMemcpyHostToDevice);

  int transpose_blocksize = 256;
  int amount_of_transpose_blocks =
     ceil(static_cast<float>(fft_length) /
          static_cast<float>(transpose_blocksize));
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  TransposeKernel<<<amount_of_transpose_blocks, transpose_blocksize>>>(
      dptr_input_RE, dptr_input_IM, dptr_results_RE, dptr_results_IM,
      fft_length, 2, 0);

  //Memcpy of input data to device
  cudaMemcpy(data_1.get(), dptr_input_RE, 2 * fft_length * sizeof(__half),
                 cudaMemcpyDeviceToHost);

  WriteResultsToFile("transposed_test_kernel.dat", fft_length, data_1.get());

  for(int i=0; i<fft_length; i++){
    float cpu_re = data_2[i];
    float gpu_re = data_1[i];
    float cpu_im = data_2[i + fft_length];
    float gpu_im = data_1[i + fft_length];
    if ((cpu_re != gpu_re) || (cpu_im != gpu_im)){
      std::cout << "Results of transpose on cpu and gpu are different!"
                << std::endl;
      return false;
    }
  }

  return true;
}

bool transpose16_2_test(){
  int fft_length = 16*16*16*2*2;

  std::vector<float> weights;
  weights.push_back(1.0);
  std::unique_ptr<__half[]> data_1 =
      CreateSineSuperpostion(fft_length, weights);
  std::unique_ptr<__half[]> data_2 =
      CreateSineSuperpostion(fft_length, weights);

  WriteResultsToFile("input.dat", fft_length, data_1.get());

  __half old1_RE[16*16*16*2][2];
  __half old1_IM[16*16*16*2][2];
  __half new1_RE[2][2*16*16*16];
  __half new1_IM[2][2*16*16*16];

  for(int i=0; i<2; i++){
    for(int j=0; j<16*16*16*2; j++){
      old1_RE[j][i] = data_2[i + 2*j];
      old1_IM[j][i] = data_2[i + 2*j + fft_length];
    }
  }

  for(int i=0; i<2; i++){
    for(int j=0; j<16*16*16*2; j++){
      new1_RE[i][j] = old1_RE[j][i];
      new1_IM[i][j] = old1_IM[j][i];
    }
  }

  __half old2_RE[2][16*16*16][2];
  __half old2_IM[2][16*16*16][2];
  __half new2_RE[2][2][16*16*16];
  __half new2_IM[2][2][16*16*16];

  for(int i=0; i<2; i++){
    for(int j=0; j<16*16*16; j++){
      for(int k=0; k<2; k++){
        old2_RE[i][j][k] = new1_RE[i][k + 2 * j];
        old2_IM[i][j][k] = new1_IM[i][k + 2 * j];
      }
    }
  }

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      for(int k=0; k<16*16*16; k++){
        new2_RE[i][j][k] = old2_RE[i][k][j];
        new2_IM[i][j][k] = old2_IM[i][k][j];
      }
    }
  }

  __half old3_RE[2][2][16*16][16];
  __half old3_IM[2][2][16*16][16];
  __half new3_RE[2][2][16][16*16];
  __half new3_IM[2][2][16][16*16];

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      for(int k=0; k<16*16; k++){
        for(int l=0; l<16; l++){
          old3_RE[i][j][k][l] = new2_RE[i][j][l + 16 * k];
          old3_IM[i][j][k][l] = new2_IM[i][j][l + 16 * k];
        }
      }
    }
  }

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      for(int k=0; k<16; k++){
        for(int l=0; l<16*16; l++){
          new3_RE[i][j][k][l] = old3_RE[i][j][l][k];
          new3_IM[i][j][k][l] = old3_IM[i][j][l][k];
        }
      }
    }
  }

  __half old4_RE[2][2][16][16][16];
  __half old4_IM[2][2][16][16][16];
  __half new4_RE[2][2][16][16][16];
  __half new4_IM[2][2][16][16][16];

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      for(int k=0; k<16; k++){
        for(int l=0; l<16; l++){
          for(int m=0; m<16; m++){
            old4_RE[i][j][k][l][m] = new3_RE[i][j][k][m + 16 * l];
            old4_IM[i][j][k][l][m] = new3_IM[i][j][k][m + 16 * l];
          }
        }
      }
    }
  }

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      for(int k=0; k<16; k++){
        for(int l=0; l<16; l++){
          for(int m=0; m<16; m++){
            new4_RE[i][j][k][l][m] = old4_RE[i][j][k][m][l];
            new4_IM[i][j][k][l][m] = old4_IM[i][j][k][m][l];
          }
        }
      }
    }
  }

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      for(int k=0; k<16; k++){
        for(int l=0; l<16; l++){
          for(int m=0; m<16; m++){
            data_2[m + 16*l + 16*16*k + 16*16*16*j + 16*16*16*2*i] =
                new4_RE[i][j][k][l][m];
            data_2[m + 16*l + 16*16*k + 16*16*16*j + 16*16*16*2*i + fft_length]
                = new4_IM[i][j][k][l][m];
          }
        }
      }
    }
  }

  WriteResultsToFile("transposed_test_cpu.dat", fft_length, data_2.get());

  __half* dptr_input_RE;
  __half* dptr_input_IM;
  __half* dptr_results_RE;
  __half* dptr_results_IM;
  cudaMalloc((void**)(&dptr_input_RE), 4 * sizeof(__half) * fft_length);

  dptr_input_IM = dptr_input_RE + fft_length;
  dptr_results_RE = dptr_input_IM + fft_length;
  dptr_results_IM = dptr_results_RE + fft_length;

  cudaMemcpy(dptr_input_RE, data_1.get(), 2 * fft_length * sizeof(__half),
             cudaMemcpyHostToDevice);

  int transpose_blocksize = 256;
  int amount_of_transpose_blocks =
     ceil(static_cast<float>(fft_length) /
          static_cast<float>(transpose_blocksize));
  //Launch kernel that performs the transposes to prepare the data for the
  //radix steps
  TransposeKernel<<<amount_of_transpose_blocks, transpose_blocksize>>>(
      dptr_input_RE, dptr_input_IM, dptr_results_RE, dptr_results_IM,
      fft_length, 2, 2);

  //Memcpy of input data to device
  cudaMemcpy(data_1.get(), dptr_input_RE, 2 * fft_length * sizeof(__half),
                 cudaMemcpyDeviceToHost);

  WriteResultsToFile("transposed_test_kernel.dat", fft_length, data_1.get());

  for(int i=0; i<fft_length; i++){
    float cpu_re = data_2[i];
    float gpu_re = data_1[i];
    float cpu_im = data_2[i + fft_length];
    float gpu_im = data_1[i + fft_length];
    if ((cpu_re != gpu_re) || (cpu_im != gpu_im)){
      std::cout << "Results of transpose on cpu and gpu are different!"
                << std::endl;
      return false;
    }
  }

  return true;
}
