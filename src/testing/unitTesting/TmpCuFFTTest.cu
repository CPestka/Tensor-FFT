#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CuFFTTest.cu"

int main(){
  std::optional<std::string> tmp = CreateComparisonData(4096, "tmp_test.dat");
}
