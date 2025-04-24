/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

// Maps the names of CUDA Device/Host types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_DEVICE_TYPE_NAME_MAP {
  // float16 Precision Device types
  {"__half",                               {"__half",                                "rocblas_half",                            CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__half_raw",                           {"__half_raw",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__half2",                              {"__half2",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__half2_raw",                          {"__half2_raw",                           "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  // Bfloat16 Precision Device types
  {"__nv_bfloat16",                        {"__hip_bfloat16",                        "rocblas_bfloat16",                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"nv_bfloat16",                          {"hip_bfloat16",                          "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_bfloat16_raw",                    {"__hip_bfloat16_raw",                    "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_bfloat162",                       {"__hip_bfloat162",                       "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"nv_bfloat162",                         {"hip_bfloat162",                         "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_bfloat162_raw",                   {"__hip_bfloat162_raw",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  // float8 Precision Device types
  {"__nv_fp8_storage_t",                   {"__hip_fp8_storage_t",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8x2_storage_t",                 {"__hip_fp8x2_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8x4_storage_t",                 {"__hip_fp8x4_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8_e5m2",                        {"__hip_fp8_e5m2_fnuz",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8x2_e5m2",                      {"__hip_fp8x2_e5m2_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8_e4m3",                        {"__hip_fp8_e4m3_fnuz",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8x2_e4m3",                      {"__hip_fp8x2_e4m3_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8x4_e4m3",                      {"__hip_fp8x4_e4m3_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_saturation_t",                    {"__hip_saturation_t",                    "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__NV_NOSAT",                           {"__HIP_NOSAT",                           "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2}},
  {"__NV_SATFINITE",                       {"__HIP_SATFINITE",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2}},
  {"__nv_fp8_interpretation_t",            {"__hip_fp8_interpretation_t",            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__NV_E4M3",                            {"__HIP_E4M3_FNUZ",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2}},
  {"__NV_E5M2",                            {"__HIP_E5M2_FNUZ",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2}},
  {"__nv_fp8x4_e5m2",                      {"__hip_fp8x4_e5m2_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2}},
  {"__nv_fp8_e8m0",                        {"__hip_fp8_e8m0",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp8x2_e8m0",                      {"__hip_fp8x2_e8m0",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp8x4_e8m0",                      {"__hip_fp8x4_e8m0",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  // float6 Precision Device types
  {"__nv_fp6_storage_t",                   {"__hip_fp6_storage_t",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6x2_storage_t",                 {"__hip_fp6x2_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6x4_storage_t",                 {"__hip_fp6x4_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6_interpretation_t",            {"__hip_fp6_interpretation_t",            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__NV_E2M3",                            {"__HIP_E2M3",                            "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, UNSUPPORTED}},
  {"__NV_E3M2",                            {"__HIP_E3M2",                            "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6_e3m2",                        {"__hip_fp6_e3m2",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6x2_e3m2",                      {"__hip_fp6x2_e3m2",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6x4_e3m2",                      {"__hip_fp6x4_e3m2",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6_e2m3",                        {"__hip_fp6_e2m3",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6x2_e2m3",                      {"__hip_fp6x2_e2m3",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp6x4_e2m3",                      {"__hip_fp6x4_e2m3",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  // float4 Precision Device types
  {"__nv_fp4_storage_t",                   {"__hip_fp4_storage_t",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"__nv_fp4x2_storage_t",                 {"__hip_fp4x2_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"__nv_fp4x4_storage_t",                 {"__hip_fp4x4_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"__nv_fp4_interpretation_t",            {"__hip_fp4_interpretation_t",            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"__NV_E2M1",                            {"__HIP_E2M1",                            "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"__nv_fp4_e2m1",                        {"__hip_fp4_e2m1",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp4x2_e2m1",                      {"__hip_fp4x2_e2m1",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  {"__nv_fp4x4_e2m1",                      {"__hip_fp4x4_e2m1",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED}},
  // defines
  {"CUDART_INF_FP16",                      {"HIPRT_INF_FP16",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"CUDART_MAX_NORMAL_FP16",               {"HIPRT_MAX_NORMAL_FP16",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"CUDART_MIN_DENORM_FP16",               {"HIPRT_MIN_DENORM_FP16",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"CUDART_NAN_FP16",                      {"HIPRT_NAN_FP16",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"CUDART_NEG_ZERO_FP16",                 {"HIPRT_NEG_ZERO_FP16",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"CUDART_ONE_FP16",                      {"HIPRT_ONE_FP16",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"CUDART_ZERO_FP16",                     {"HIPRT_ZERO_FP16",                       "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  // builtins
  {"cudaRoundMode",                        {"hipRoundMode",                          "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"cudaRoundNearest",                     {"hipRoundNearest",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"cudaRoundZero",                        {"hipRoundZero",                          "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"cudaRoundPosInf",                      {"hipRoundPosInf",                        "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
  {"cudaRoundMinInf",                      {"hipRoundMinInf",                        "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2, HIP_EXPERIMENTAL}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DEVICE_TYPE_NAME_VER_MAP {
  {"__nv_bfloat16",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"nv_bfloat16",                          {CUDA_110, CUDA_0,   CUDA_0  }},
  {"__nv_bfloat16_raw",                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"__nv_bfloat162",                       {CUDA_110, CUDA_0,   CUDA_0  }},
  {"nv_bfloat162",                         {CUDA_110, CUDA_0,   CUDA_0  }},
  {"__nv_bfloat162_raw",                   {CUDA_110, CUDA_0,   CUDA_0  }},
  {"__nv_fp8_storage_t",                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x2_storage_t",                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x4_storage_t",                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8_e5m2",                        {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x2_e5m2",                      {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8_e4m3",                        {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x2_e4m3",                      {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x4_e4m3",                      {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_saturation_t",                    {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__NV_NOSAT",                           {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__NV_SATFINITE",                       {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8_interpretation_t",            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__NV_E4M3",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__NV_E5M2",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x4_e5m2",                      {CUDA_118, CUDA_0,   CUDA_0  }},
  {"__nv_fp6_storage_t",                   {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6x2_storage_t",                 {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6x4_storage_t",                 {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6_interpretation_t",            {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__NV_E2M3",                            {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__NV_E3M2",                            {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6_e3m2",                        {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6x2_e3m2",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6x4_e3m2",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6_e2m3",                        {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6x2_e2m3",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp6x4_e2m3",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4_storage_t",                   {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4x2_storage_t",                 {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4x4_storage_t",                 {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4_interpretation_t",            {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__NV_E2M1",                            {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4_e2m1",                        {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4x2_e2m1",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp4x4_e2m1",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp8_e8m0",                        {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x2_e8m0",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"__nv_fp8x4_e8m0",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"CUDART_INF_FP16",                      {CUDA_122, CUDA_0,   CUDA_0  }},
  {"CUDART_MAX_NORMAL_FP16",               {CUDA_122, CUDA_0,   CUDA_0  }},
  {"CUDART_MIN_DENORM_FP16",               {CUDA_122, CUDA_0,   CUDA_0  }},
  {"CUDART_NAN_FP16",                      {CUDA_122, CUDA_0,   CUDA_0  }},
  {"CUDART_NEG_ZERO_FP16",                 {CUDA_122, CUDA_0,   CUDA_0  }},
  {"CUDART_ONE_FP16",                      {CUDA_122, CUDA_0,   CUDA_0  }},
  {"CUDART_ZERO_FP16",                     {CUDA_122, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_DEVICE_TYPE_NAME_VER_MAP {
  {"__half",                               {HIP_1060, HIP_0,    HIP_0   }},
  {"__half2",                              {HIP_1060, HIP_0,    HIP_0   }},
  {"__half_raw",                           {HIP_1090, HIP_0,    HIP_0   }},
  {"__half2_raw",                          {HIP_1090, HIP_0,    HIP_0   }},
  {"__hip_bfloat16",                       {HIP_5070, HIP_0,    HIP_0   }},
  {"__hip_fp8_e4m3_fnuz",                  {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8_storage_t",                  {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8x2_storage_t",                {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8x4_storage_t",                {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8_e5m2_fnuz",                  {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8x2_e5m2_fnuz",                {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8x2_e4m3_fnuz",                {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8x4_e4m3_fnuz",                {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_saturation_t",                   {HIP_6020, HIP_0,    HIP_0   }},
  {"__HIP_NOSAT",                          {HIP_6020, HIP_0,    HIP_0   }},
  {"__HIP_SATFINITE",                      {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8_interpretation_t",           {HIP_6020, HIP_0,    HIP_0   }},
  {"__HIP_E4M3_FNUZ",                      {HIP_6020, HIP_0,    HIP_0   }},
  {"__HIP_E5M2_FNUZ",                      {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_fp8x4_e5m2_fnuz",                {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_bfloat16_raw",                   {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_bfloat162_raw",                  {HIP_6020, HIP_0,    HIP_0   }},
  {"__hip_bfloat162",                      {HIP_5070, HIP_0,    HIP_0   }},
  {"hip_bfloat16",                         {HIP_3050, HIP_0,    HIP_0   }},
  {"HIPRT_INF_FP16",                       {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPRT_MAX_NORMAL_FP16",                {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPRT_MIN_DENORM_FP16",                {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPRT_NAN_FP16",                       {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPRT_NEG_ZERO_FP16",                  {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPRT_ONE_FP16",                       {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPRT_ZERO_FP16",                      {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"__hip_fp4_storage_t",                  {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"__hip_fp4x2_storage_t",                {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"__hip_fp4x4_storage_t",                {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"__hip_fp4_interpretation_t",           {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"__HIP_E2M1",                           {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipRoundMode",                         {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipRoundNearest",                      {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipRoundZero",                         {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipRoundPosInf",                       {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipRoundMinInf",                       {HIP_6050, HIP_0,    HIP_0,  HIP_LATEST}},

  {"rocblas_half",                         {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_bfloat16",                     {HIP_3050, HIP_0,    HIP_0   }},
};
