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

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_TENSOR_TYPE_NAME_MAP {
  // cuTENSOR defines


  // cuTENSOR enums
    {"cutensorDataType_t",                               {"",                                                         "",           CONV_TYPE, API_TENSOR, 1, HIP_UNSUPPORTED}},
    {"cutensorOperator_t",                               {"",                                                         "",           CONV_TYPE, API_TENSOR, 1, HIP_UNSUPPORTED}},

    {"cutensorStatus_t",                                 {"hiptensorStatus_t",                                        "",           CONV_TYPE, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_SUCCESS",                          {"HIPTENSOR_STATUS_SUCCESS",                                 "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_NOT_INITIALIZED",                  {"HIPTENSOR_STATUS_NOT_INITIALIZED",                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_ALLOC_FAILED",                     {"HIPTENSOR_STATUS_ALLOC_FAILED",                            "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_INVALID_VALUE",                    {"HIPTENSOR_STATUS_INVALID_VALUE",                           "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_ARCH_MISMATCH",                    {"HIPTENSOR_STATUS_ARCH_MISMATCH",                           "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_MAPPING_ERROR",                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_EXECUTION_FAILED",                 {"HIPTENSOR_STATUS_EXECUTION_FAILED",                        "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_INTERNAL_ERROR",                   {"HIPTENSOR_STATUS_INTERNAL_ERROR",                          "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_NOT_SUPPORTED",                    {"HIPTENSOR_STATUS_NOT_SUPPORTED",                           "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_LICENSE_ERROR",                    {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_CUBLAS_ERROR",                     {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_CUDA_ERROR",                       {"",                                                         "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE",           {"HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE",                  "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_INSUFFICIENT_DRIVER",              {"HIPTENSOR_STATUS_INSUFFICIENT_DRIVER",                     "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
    {"CUTENSOR_STATUS_IO_ERROR",                         {"HIPTENSOR_STATUS_IO_ERROR",                                "",           CONV_NUMERIC_LITERAL, API_TENSOR, 1}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_TENSOR_TYPE_NAME_VER_MAP {
};
