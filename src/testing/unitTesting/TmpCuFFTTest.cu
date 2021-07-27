#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CuFFTTest.cu"

int main(){
  std::optional<std::string> tmp =
      CreateComparisonDataHalf(4096, "tmp_test_1.dat");
  tmp = CreateComparisonDataDouble(4096, "tmp_test_2.dat");
}
