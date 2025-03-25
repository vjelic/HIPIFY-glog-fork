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

  // CHECK: hipfftResult fftResult;
  cufftResult fftResult;

  // CHECK: hipfftHandle fftHandle;
  cufftHandle fftHandle;

  // CHECK: hipfftXtSubFormat fftXtSubFormat;
  cufftXtSubFormat fftXtSubFormat;

  // CHECK: hipfftXtCopyType fftXtCopyType;
  cufftXtCopyType fftXtCopyType;

  // CHECK: hipLibXtDesc *descptr = nullptr;
  // CHECK-NEXT: hipLibXtDesc *input_desc = nullptr;
  // CHECK-NEXT: hipLibXtDesc *output_desc = nullptr;
  cudaLibXtDesc *descptr = nullptr;
  cudaLibXtDesc *input_desc = nullptr;
  cudaLibXtDesc *output_desc = nullptr;

  // CHECK: hipLibXtDesc **desc = nullptr;
  cudaLibXtDesc **desc = nullptr;

  int *gpu = nullptr;
  int count = 0;
  void *dstptr = nullptr;
  void *srcptr = nullptr;
  int dir = 0;
  int rank = 0;
  long long int *n = nullptr;
  long long int *inembed = nullptr;
  long long int istride = 0;
  long long int idist = 0;
  long long int *onembed = nullptr;
  long long int ostride = 0;
  long long int odist = 0;
  long long int batch = 0;
  size_t *workSize = nullptr;
  void *input = nullptr;
  void *output = nullptr;

  // CHECK: hipDataType executionType;
  // CHECK-NEXT: hipDataType inputType;
  // CHECK-NEXT: hipDataType outputType;
  cudaDataType executionType;
  cudaDataType inputType;
  cudaDataType outputType;

  // CUDA: cufftResult CUFFTAPI cufftXtSetGPUs(cufftHandle handle, int nGPUs, int *whichGPUs);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtSetGPUs(hipfftHandle plan, int count, int* gpus);
  // CHECK: fftResult = hipfftXtSetGPUs(fftHandle, count, gpu);
  fftResult = cufftXtSetGPUs(fftHandle, count, gpu);

  // CUDA: cufftResult CUFFTAPI cufftXtMalloc(cufftHandle plan, cudaLibXtDesc ** descriptor, cufftXtSubFormat format);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtMalloc(hipfftHandle plan, hipLibXtDesc** desc, hipfftXtSubFormat format);
  // CHECK: fftResult = hipfftXtMalloc(fftHandle, desc, fftXtSubFormat);
  fftResult = cufftXtMalloc(fftHandle, desc, fftXtSubFormat);

  // CUDA: cufftResult CUFFTAPI cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer, cufftXtCopyType type);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtMemcpy(hipfftHandle plan, void* dest, void* src, hipfftXtCopyType type);
  // CHECK: fftResult = hipfftXtMemcpy(fftHandle, dstptr, srcptr, fftXtCopyType);
  fftResult = cufftXtMemcpy(fftHandle, dstptr, srcptr, fftXtCopyType);

  // CUDA: cufftResult CUFFTAPI cufftXtFree(cudaLibXtDesc *descriptor);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtFree(hipLibXtDesc* desc);
  // CHECK: fftResult = hipfftXtFree(descptr);
  fftResult = cufftXtFree(descptr);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output, int direction);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorC2C(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output, int direction);
  // CHECK: fftResult = hipfftXtExecDescriptorC2C(fftHandle, input_desc, output_desc, dir);
  fftResult = cufftXtExecDescriptorC2C(fftHandle, input_desc, output_desc, dir);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorR2C(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output);
  // CHECK: fftResult = hipfftXtExecDescriptorR2C(fftHandle, input_desc, output_desc);
  fftResult = cufftXtExecDescriptorR2C(fftHandle, input_desc, output_desc);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorC2R(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output);
  // CHECK: fftResult = hipfftXtExecDescriptorC2R(fftHandle, input_desc, output_desc);
  fftResult = cufftXtExecDescriptorC2R(fftHandle, input_desc, output_desc);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output, int direction);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorZ2Z(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output, int direction);
  // CHECK: fftResult = hipfftXtExecDescriptorZ2Z(fftHandle, input_desc, output_desc, dir);
  fftResult = cufftXtExecDescriptorZ2Z(fftHandle, input_desc, output_desc, dir);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorD2Z(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output);
  // CHECK: fftResult = hipfftXtExecDescriptorD2Z(fftHandle, input_desc, output_desc);
  fftResult = cufftXtExecDescriptorD2Z(fftHandle, input_desc, output_desc);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptorZ2D(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output);
  // CHECK: fftResult = hipfftXtExecDescriptorZ2D(fftHandle, input_desc, output_desc);
  fftResult = cufftXtExecDescriptorZ2D(fftHandle, input_desc, output_desc);

#if CUDA_VERSION >= 8000
  // CUDA: cufftResult CUFFTAPI cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int *onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t *workSize, cudaDataType executiontype);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtMakePlanMany(hipfftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, hipDataType inputType, long long int* onembed, long long int ostride, long long int odist, hipDataType outputType, long long int batch, size_t* workSize, hipDataType executionType);
  // CHECK: fftResult = hipfftXtMakePlanMany(fftHandle, rank, n, inembed, istride, idist, inputType, onembed, ostride, odist, outputType, batch, workSize, executionType);
  fftResult = cufftXtMakePlanMany(fftHandle, rank, n, inembed, istride, idist, inputType, onembed, ostride, odist, outputType, batch, workSize, executionType);

  // CUDA: cufftResult CUFFTAPI cufftXtGetSizeMany(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int *onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t *workSize, cudaDataType executiontype);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtGetSizeMany(hipfftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, hipDataType inputType, long long int* onembed, long long int ostride, long long int odist, hipDataType outputType, long long int batch, size_t* workSize, hipDataType executionType);
  // CHECK: fftResult = hipfftXtGetSizeMany(fftHandle, rank, n, inembed, istride, idist, inputType, onembed, ostride, odist, outputType, batch, workSize, executionType);
  fftResult = cufftXtGetSizeMany(fftHandle, rank, n, inembed, istride, idist, inputType, onembed, ostride, odist, outputType, batch, workSize, executionType);

  // CUDA: cufftResult CUFFTAPI cufftXtExec(cufftHandle plan, void *input, void *output, int direction);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExec(hipfftHandle plan, void* input, void* output, int direction);
  // CHECK: fftResult = hipfftXtExec(fftHandle, input, output, dir);
  fftResult = cufftXtExec(fftHandle, input, output, dir);

  // CUDA: cufftResult CUFFTAPI cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output, int direction);
  // HIP: HIPFFT_EXPORT hipfftResult hipfftXtExecDescriptor(hipfftHandle plan, hipLibXtDesc* input, hipLibXtDesc* output, int direction);
  // CHECK: fftResult = hipfftXtExecDescriptor(fftHandle, input_desc, output_desc, dir);
  fftResult = cufftXtExecDescriptor(fftHandle, input_desc, output_desc, dir);
#endif

  return 0;
}
