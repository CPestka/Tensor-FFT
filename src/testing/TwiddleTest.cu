//Util to test accuracy of device functions 
#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void ComputeTwiddle(__half* output, const int length_halfed){
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  for(int k=0; k<16; k++){
    int matrix_id = k + 16 * thread_id;

    output[matrix_id] =
        static_cast<__half>(cos((M_PI * thread_id * k)/length_halfed));
    output[matrix_id + 32 * length_halfed] =
        static_cast<__half>(-sin((M_PI * thread_id * k)/length_halfed));

    output[matrix_id + 64 * length_halfed] =
        static_cast<__half>(cosf((M_PI * thread_id * k)/length_halfed));
    output[matrix_id + 64 * length_halfed + 32 * length_halfed] =
        static_cast<__half>(-sinf((M_PI * thread_id * k)/length_halfed));

    output[matrix_id + 128 * length_halfed] =
        hcos(__hdiv(__hmul(static_cast<__half>(M_PI),
                           static_cast<__half>(thread_id * k)),
                    static_cast<__half>(length_halfed)));
    output[matrix_id + 128 * length_halfed + 32 * length_halfed] =
        hcos(__hdiv(__hmul(static_cast<__half>(M_PI),
                           static_cast<__half>(thread_id * k)),
                    static_cast<__half>(length_halfed)));
  }
}

int main() {
  constexpr int n = 16;
  constexpr int m = 16;

  std::unique_ptr<__half[]> results = std::make_unique<__half[]>(2 * 3 * n * m);

  __half* dptr_results;
  cudaMalloc((void**)(&dptr_results), 3 * sizeof(__half) * 2 * n * m);

  ComputeTwiddle<<< 1, 16 >>>(dptr_results, m/2);

  cudaDeviceSynchronize();

  cudaMemcpy(results.get(), dptr_results, 3 * sizeof(__half) * 2 * n * m,
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  /*
  for(int i=0; i<n*m; i++){
    double fp32_trig_RE = results[i];
    double fp32_trig_IM = results[i + (n*m)];

    double fast_trig_RE = results[i + (2*n*m)];
    double fast_trig_IM = results[i + (3*n*m)];

    double half_trig_RE = results[i + (4*n*m)];
    double half_trig_IM = results[i + (5*n*m)];

    if (fp32_trig_RE != fast_trig_RE) {
      std::cout << "1_RE" << std::endl;
    }
    if (fp32_trig_IM != fast_trig_IM) {
      std::cout << "1_IM" << std::endl;
    }

    if (fp32_trig_RE != half_trig_RE) {
      std::cout << "2_RE" << std::endl;
    }
    if (fp32_trig_IM != half_trig_IM) {
      std::cout << "2_IM" << std::endl;
    }
  }
  */

  std::cout << "cos(): RE" << std::endl;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      std::cout << static_cast<double>(results[j + (i * m)]) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "cos(): IM" << std::endl;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      std::cout << static_cast<double>(results[j + (i * m) + (n*m)]) << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "cosf(): RE" << std::endl;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      std::cout << static_cast<double>(results[j + (i * m) + (2*n*m)]) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "cosf(): IM" << std::endl;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      std::cout << static_cast<double>(results[j + (i * m) + (3*n*m)]) << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "hcos(): RE" << std::endl;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      std::cout << static_cast<double>(results[j + (i * m) + (4*n*m)]) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "hcos(): IM" << std::endl;
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      std::cout << static_cast<double>(results[j + (i * m) + (5*n*m)]) << "\t";
    }
    std::cout << std::endl;
  }
}
