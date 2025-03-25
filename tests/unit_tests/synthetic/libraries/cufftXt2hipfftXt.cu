// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipfft/hipfftXt.h"
#include "cufftXt.h"
// CHECK-NOT: #include "hipfftXt.h"

int main() {
  printf("25. cufftXt API to hipfftXt API synthetic test\n");

  // CHECK: hipfftXtSubFormat_t fftXtSubFormat_t;
  // CHECK-NEXT: hipfftXtSubFormat_t FFT_XT_FORMAT_INPUT = HIPFFT_XT_FORMAT_INPUT;
  // CHECK-NEXT: hipfftXtSubFormat_t FFT_XT_FORMAT_OUTPUT = HIPFFT_XT_FORMAT_OUTPUT;
  // CHECK-NEXT: hipfftXtSubFormat_t FFT_XT_FORMAT_INPLACE = HIPFFT_XT_FORMAT_INPLACE;
  // CHECK-NEXT: hipfftXtSubFormat_t FFT_XT_FORMAT_INPLACE_SHUFFLED = HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
  // CHECK-NEXT: hipfftXtSubFormat_t FFT_XT_FORMAT_1D_INPUT_SHUFFLED = HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
  // CHECK-NEXT: hipfftXtSubFormat_t FFT_FORMAT_UNDEFINED = HIPFFT_FORMAT_UNDEFINED;
  cufftXtSubFormat_t fftXtSubFormat_t;
  cufftXtSubFormat_t FFT_XT_FORMAT_INPUT = CUFFT_XT_FORMAT_INPUT;
  cufftXtSubFormat_t FFT_XT_FORMAT_OUTPUT = CUFFT_XT_FORMAT_OUTPUT;
  cufftXtSubFormat_t FFT_XT_FORMAT_INPLACE = CUFFT_XT_FORMAT_INPLACE;
  cufftXtSubFormat_t FFT_XT_FORMAT_INPLACE_SHUFFLED = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
  cufftXtSubFormat_t FFT_XT_FORMAT_1D_INPUT_SHUFFLED = CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
  cufftXtSubFormat_t FFT_FORMAT_UNDEFINED = CUFFT_FORMAT_UNDEFINED;

  // CHECK: hipfftXtCopyType_t fftXtCopyType_t;
  // CHECK-NEXT: hipfftXtCopyType_t FFT_COPY_HOST_TO_DEVICE = HIPFFT_COPY_HOST_TO_DEVICE;
  // CHECK-NEXT: hipfftXtCopyType_t FFT_COPY_DEVICE_TO_HOST = HIPFFT_COPY_DEVICE_TO_HOST;
  // CHECK-NEXT: hipfftXtCopyType_t FFT_COPY_DEVICE_TO_DEVICE = HIPFFT_COPY_DEVICE_TO_DEVICE;
  // CHECK-NEXT: hipfftXtCopyType_t FFT_COPY_UNDEFINED = HIPFFT_COPY_UNDEFINED;
  cufftXtCopyType_t fftXtCopyType_t;
  cufftXtCopyType_t FFT_COPY_HOST_TO_DEVICE = CUFFT_COPY_HOST_TO_DEVICE;
  cufftXtCopyType_t FFT_COPY_DEVICE_TO_HOST = CUFFT_COPY_DEVICE_TO_HOST;
  cufftXtCopyType_t FFT_COPY_DEVICE_TO_DEVICE = CUFFT_COPY_DEVICE_TO_DEVICE;
  cufftXtCopyType_t FFT_COPY_UNDEFINED = CUFFT_COPY_UNDEFINED;

  // CHECK: hipfftXtCallbackType_t fftXtCallbackType_t;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_LD_COMPLEX = HIPFFT_CB_LD_COMPLEX;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_LD_COMPLEX_DOUBLE = HIPFFT_CB_LD_COMPLEX_DOUBLE;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_LD_REAL = HIPFFT_CB_LD_REAL;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_LD_REAL_DOUBLE = HIPFFT_CB_LD_REAL_DOUBLE;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_ST_COMPLEX = HIPFFT_CB_ST_COMPLEX;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_ST_COMPLEX_DOUBLE = HIPFFT_CB_ST_COMPLEX_DOUBLE;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_ST_REAL = HIPFFT_CB_ST_REAL;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_ST_REAL_DOUBLE = HIPFFT_CB_ST_REAL_DOUBLE;
  // CHECK-NEXT: hipfftXtCallbackType_t FFT_CB_UNDEFINED = HIPFFT_CB_UNDEFINED;
  cufftXtCallbackType_t fftXtCallbackType_t;
  cufftXtCallbackType_t FFT_CB_LD_COMPLEX = CUFFT_CB_LD_COMPLEX;
  cufftXtCallbackType_t FFT_CB_LD_COMPLEX_DOUBLE = CUFFT_CB_LD_COMPLEX_DOUBLE;
  cufftXtCallbackType_t FFT_CB_LD_REAL = CUFFT_CB_LD_REAL;
  cufftXtCallbackType_t FFT_CB_LD_REAL_DOUBLE = CUFFT_CB_LD_REAL_DOUBLE;
  cufftXtCallbackType_t FFT_CB_ST_COMPLEX = CUFFT_CB_ST_COMPLEX;
  cufftXtCallbackType_t FFT_CB_ST_COMPLEX_DOUBLE = CUFFT_CB_ST_COMPLEX_DOUBLE;
  cufftXtCallbackType_t FFT_CB_ST_REAL = CUFFT_CB_ST_REAL;
  cufftXtCallbackType_t FFT_CB_ST_REAL_DOUBLE = CUFFT_CB_ST_REAL_DOUBLE;
  cufftXtCallbackType_t FFT_CB_UNDEFINED = CUFFT_CB_UNDEFINED;
  return 0;
}
