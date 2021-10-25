//Used to compute different error metrics comparing our implementation to
//different CuFFT once.

#include <memory>

#include "ComparisonDataCuFFT.cu"
#include "ErrorUtil.h"

struct Errors{
  double MaxDiv;
  double MeanAbsoluteError;
  double RootMeanSquareError;
};

struct ErrorTestDataPoint{
  int64_t fft_length;
  int amount_of_frequencies;
  double max_value;
  Errors divs;
};

template<typename Integer, typename float2_t1, typename float2_t2>
Errors ComputeErrors(float2_t1* data_1, float2_t2* data_2, Integer fft_length){
  Errors results;
  results.MaxDiv = MaxDiv(data_1, data_2, fft_length);
  results.MeanAbsoluteError = MeanAbsoluteError(data_1, data_2, fft_length);
  results.RootMeanSquareError = RootMeanSquareError(data_1, data_2, fft_length);

  return results;
}

template<typename Integer>
Errors ComputeOurVsFp64Errors(long long fft_length,
                              float2* dptr_weights,
                              int amount_of_frequencies,
                              double normalization_factor){
  std::unique_ptr<cufftDoubleComplex[]> fp64_results =
      GetComparisionFP64Data(dptr_weights, amount_of_frequencies, fft_length,
                             normalization_factor);

  std::unique_ptr<__half2[]> our_results =
       GetOurFP16Data<Integer>(dptr_weights, amount_of_frequencies, fft_length,
                               normalization_factor);

  return ComputeErrors<long long,cufftDoubleComplex,__half2>(
        fp64_results.get(), our_results.get(), fft_length);
}

Errors ComputeFP32VsFp64Errors(long long fft_length,
                               float2* dptr_weights,
                               int amount_of_frequencies,
                               double normalization_factor){
  std::unique_ptr<cufftDoubleComplex[]> fp64_results =
      GetComparisionFP64Data(dptr_weights, amount_of_frequencies, fft_length,
                             normalization_factor);

  std::unique_ptr<cufftComplex[]> fp32_results =
      GetComparisionFP32Data(dptr_weights, amount_of_frequencies, fft_length,
                             normalization_factor);

  return ComputeErrors<long long,cufftDoubleComplex,cufftComplex>(
        fp64_results.get(), fp32_results.get(), fft_length);
}

Errors ComputeFP16VsFp64Errors(long long fft_length,
                               float2* dptr_weights,
                               int amount_of_frequencies,
                               double normalization_factor){
  std::unique_ptr<cufftDoubleComplex[]> fp64_results =
      GetComparisionFP64Data(dptr_weights, amount_of_frequencies, fft_length,
                             normalization_factor);

  std::unique_ptr<__half2[]> fp16_results =
      GetComparisionFP16Data(dptr_weights, amount_of_frequencies, fft_length,
                             normalization_factor);

  return ComputeErrors<long long,cufftDoubleComplex,__half2>(
        fp64_results.get(), fp16_results.get(), fft_length);
}
