/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_TENSOR_TYPE_NAME_MAP {
  // cuTENSOR enums
  {"cutensorDataType_t",                               {"hiptensorComputeType_t",                                   "",           CONV_TYPE, API_TENSOR, 1}},
  {"CUTENSOR_R_16F",                                   {"HIPTENSOR_COMPUTE_16F",                                    "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_16F",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_16BF",                                  {"HIPTENSOR_COMPUTE_16BF",                                   "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_16BF",                                  {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_32F",                                   {"HIPTENSOR_COMPUTE_32F",                                    "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_32F",                                   {"HIPTENSOR_COMPUTE_C32F",                                   "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_R_64F",                                   {"HIPTENSOR_COMPUTE_64F",                                    "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_64F",                                   {"HIPTENSOR_COMPUTE_C64F",                                   "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_R_4I",                                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_C_4I",                                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_4U",                                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_C_4U",                                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_8I",                                    {"HIPTENSOR_COMPUTE_8I",                                     "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_8I",                                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_8U",                                    {"HIPTENSOR_COMPUTE_8U",                                     "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_8U",                                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_16I",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_C_16I",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_16U",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_C_16U",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_32I",                                   {"HIPTENSOR_COMPUTE_32I",                                    "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_32I",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_32U",                                   {"HIPTENSOR_COMPUTE_32U",                                    "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_C_32U",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_64I",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_C_64I",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_R_64U",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_C_64U",                                   {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},

  {"cutensorOperator_t",                               {"",                                                         "",           CONV_TYPE, API_TENSOR, 1, UNSUPPORTED}},

  {"cutensorStatus_t",                                 {"hiptensorStatus_t",                                        "",           CONV_TYPE, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_SUCCESS",                          {"HIPTENSOR_STATUS_SUCCESS",                                 "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_NOT_INITIALIZED",                  {"HIPTENSOR_STATUS_NOT_INITIALIZED",                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_ALLOC_FAILED",                     {"HIPTENSOR_STATUS_ALLOC_FAILED",                            "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_INVALID_VALUE",                    {"HIPTENSOR_STATUS_INVALID_VALUE",                           "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_ARCH_MISMATCH",                    {"HIPTENSOR_STATUS_ARCH_MISMATCH",                           "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_MAPPING_ERROR",                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_STATUS_EXECUTION_FAILED",                 {"HIPTENSOR_STATUS_EXECUTION_FAILED",                        "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_INTERNAL_ERROR",                   {"HIPTENSOR_STATUS_INTERNAL_ERROR",                          "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_NOT_SUPPORTED",                    {"HIPTENSOR_STATUS_NOT_SUPPORTED",                           "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_LICENSE_ERROR",                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_STATUS_CUBLAS_ERROR",                     {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_STATUS_CUDA_ERROR",                       {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1, UNSUPPORTED}},
  {"CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE",           {"HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE",                  "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_INSUFFICIENT_DRIVER",              {"HIPTENSOR_STATUS_INSUFFICIENT_DRIVER",                     "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
  {"CUTENSOR_STATUS_IO_ERROR",                         {"HIPTENSOR_STATUS_IO_ERROR",                                "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_TENSOR_TYPE_NAME_VER_MAP {
  {"cutensorDataType_t",                               {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_16F",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_16F",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_16BF",                                  {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_16BF",                                  {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_32F",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_32F",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_64F",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_64F",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_4I",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_4I",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_4U",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_4U",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_8I",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_8I",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_8U",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_8U",                                    {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_16I",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_16I",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_16U",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_16U",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_32I",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_32I",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_32U",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_32U",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_64I",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_64I",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_R_64U",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
  {"CUTENSOR_C_64U",                                   {CUTENSOR_2000,  CUDA_0,        CUDA_0,      }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_TENSOR_TYPE_NAME_VER_MAP {
  {"hiptensorComputeType_t",                           {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_16F",                            {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_16BF",                           {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_32F",                            {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_C32F",                           {HIP_6010,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_64F",                            {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_C64F",                           {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_8I",                             {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_8U",                             {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_32I",                            {HIP_5070,       HIP_0,         HIP_0,       }},
  {"HIPTENSOR_COMPUTE_32U",                            {HIP_5070,       HIP_0,         HIP_0,       }},
};
