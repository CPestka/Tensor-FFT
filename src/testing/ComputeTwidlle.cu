#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

int main() {
  constexpr int n = 16;
  constexpr int m = 16;

  std::unique_ptr<__half[]> twiddle = std::make_unique<__half[]>(2*n*m);

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      double phase = (2 * M_PI * j * i) / n;
      double re = cos(phase);
      double im = -sin(phase);
      twiddle[j + (m * i)] = static_cast<__half>(re);
      twiddle[j + (m * i) + (m * n)] = static_cast<__half>(im);
    }
  }

  std::string file_name =
      (((("twiddle_" + std::to_string(n)) + "_") + std::to_string(m)) + ".dat");
  std::ofstream myfile (file_name);
  if (myfile.is_open()) {
    myfile << std::setprecision(20);
    for(int j=0; j<n; j++){
      for(int i=0; i<m; i++){
        myfile << static_cast<double>(twiddle[i + (j * m)]) << ", ";
      }
      myfile << "\n";
    }

    myfile << "\n";
    for(int j=0; j<n; j++){
      for(int i=0; i<m; i++){
        myfile << static_cast<double>(twiddle[i + (j * m) + (m * n)]) << ", ";
      }
      myfile << "\n";
    }

    myfile.close();
  }
}
