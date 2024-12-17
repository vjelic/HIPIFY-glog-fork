// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hip/hip_fp8.h"
#include "cuda_fp8.h"
// CHECK-NOT: #include "hip/hip_fp8.h"
// CHECK-NOT: #include "cuda_fp8.h"

int main() {
  printf("24.before_11080_after_12011, CUDA Device API to HIP Device API synthetic test\n");

  double da = 0.0f;
  double dx = 0.0f;
  float fa = 0.0f;
  float fx = 0.0f;
  double2 d2 = { 0.0f, 0.0f };
  float2 f2 = { 0.0f, 0.0f };
  __half_raw hrx = { 0 };
  __half2_raw h2rx = { 0, 0 };

#if CUDA_VERSION >= 11000
  // CHECK: __hip_bfloat16 bf16 = { 0 };
  __nv_bfloat16 bf16 = { 0 };
  // CHECK: __hip_bfloat162 bf162 = { 0, 0 };
  // CHECK-NEXT: __hip_bfloat162 bf162a = { 0, 0 };
  // CHECK-NEXT: __hip_bfloat162 bf162b = { 0, 0 };
  __nv_bfloat162 bf162 = { 0, 0 };
  __nv_bfloat162 bf162a = { 0, 0 };
  __nv_bfloat162 bf162b = { 0, 0 };

#if CUDA_VERSION < 11080 || CUDA_VERSION >= 12000
  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __bfloat162bfloat162(const __nv_bfloat16 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __bfloat162bfloat162(const __hip_bfloat16 a);
  // CHECK: bf162 = __bfloat162bfloat162(bf16);
  bf162 = __bfloat162bfloat162(bf16);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __lows2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lows2bfloat162(const __hip_bfloat162 a, const __hip_bfloat162 b);
  // CHECK: bf162 = __lows2bfloat162(bf162a, bf162b);
  bf162 = __lows2bfloat162(bf162a, bf162b);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __highs2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __highs2bfloat162(const __hip_bfloat162 a, const __hip_bfloat162 b);
  // CHECK: bf162 = __highs2bfloat162(bf162a, bf162b);
  bf162 = __highs2bfloat162(bf162a, bf162b);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __high2bfloat16(const __nv_bfloat162 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __high2bfloat16(const __hip_bfloat162 a);
  // CHECK: bf16 = __high2bfloat16(bf162a);
  bf16 = __high2bfloat16(bf162a);
#endif

#endif

  return 0;
}
