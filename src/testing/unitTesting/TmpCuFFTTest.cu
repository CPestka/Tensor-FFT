#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CuFFTTest.cu"

int main(){
  CreateComparisonData(4096, "tmp_test.dat");
}
