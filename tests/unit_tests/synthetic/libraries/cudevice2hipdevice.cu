// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hip/hip_fp8.h"
#include "cuda_fp8.h"
// CHECK-NOT: #include "hip/hip_fp8.h"
// CHECK-NOT: #include "cuda_fp8.h"

int main() {
  printf("24. CUDA Device API to HIP Device API synthetic test\n");

#if CUDA_VERSION >= 11080
  // CHECK: __hip_fp8_storage_t fp8_storage_t;
  __nv_fp8_storage_t fp8_storage_t;

  // CHECK: __hip_fp8x2_storage_t fp8x2_storage_t;
  __nv_fp8x2_storage_t fp8x2_storage_t;

  // CHECK: __hip_fp8x4_storage_t fp8x4_storage_t;
  __nv_fp8x4_storage_t fp8x4_storage_t;

  // CHECK: __hip_fp8_e5m2_fnuz fp8_e5m2;
  __nv_fp8_e5m2 fp8_e5m2;

  // CHECK: __hip_fp8x2_e5m2_fnuz fp8x2_e5m2;
  __nv_fp8x2_e5m2 fp8x2_e5m2;

  // CHECK: __hip_fp8_e4m3_fnuz fp8_e4m3;
  __nv_fp8_e4m3 fp8_e4m3;

  // CHECK: __hip_fp8x2_e4m3_fnuz fp8x2_e4m3;
  __nv_fp8x2_e4m3 fp8x2_e4m3;

  // CHECK: __hip_fp8x4_e4m3_fnuz fp8x4_e4m3;
  __nv_fp8x4_e4m3 fp8x4_e4m3;

  // CHECK: __hip_saturation_t saturation_t;
  // CHECK-NEXT: __hip_saturation_t NOSAT = __HIP_NOSAT;
  // CHECK-NEXT: __hip_saturation_t SATFINITE = __HIP_SATFINITE;
  __nv_saturation_t saturation_t;
  __nv_saturation_t NOSAT = __NV_NOSAT;
  __nv_saturation_t SATFINITE = __NV_SATFINITE;

  // CHECK: __hip_fp8_interpretation_t fp8_interpretation_t;
  // CHECK-NEXT: __hip_fp8_interpretation_t E4M3 = __HIP_E4M3_FNUZ;
  // CHECK-NEXT: __hip_fp8_interpretation_t E5M2 = __HIP_E5M2_FNUZ;
  __nv_fp8_interpretation_t fp8_interpretation_t;
  __nv_fp8_interpretation_t E4M3 = __NV_E4M3;
  __nv_fp8_interpretation_t E5M2 = __NV_E5M2;

  // CHECK: __hip_fp8x4_e5m2_fnuz fp8x4_e5m2;
  __nv_fp8x4_e5m2 fp8x4_e5m2;
#endif

  return 0;
}
