# CUTENSOR API supported by HIP

## **1. CUTENSOR Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUTENSOR_C_16BF`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_16F`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_16I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_16U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_32F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_C32F`|6.1.0| | | | |
|`CUTENSOR_C_32I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_32U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_4I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_4U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_64F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_C64F`|5.7.0| | | | |
|`CUTENSOR_C_64I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_64U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_8I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_C_8U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_ABS`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ACOS`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ACOSH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ADD`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ASIN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ASINH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ATAN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_ATANH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_CEIL`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_CONJ`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_COS`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_COSH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_EXP`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_FLOOR`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_IDENTITY`|1.0.1.0| | | |`HIPTENSOR_OP_IDENTITY`|5.7.0| | | | |
|`CUTENSOR_OP_LOG`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_MAX`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_MIN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_MISH`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_MUL`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_NEG`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_RCP`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_RELU`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SIGMOID`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SIN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SINH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_SOFT_PLUS`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_SOFT_SIGN`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_SQRT`|1.0.1.0| | | |`HIPTENSOR_OP_SQRT`|6.2.0| | | | |
|`CUTENSOR_OP_SWISH`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_OP_TAN`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_TANH`|1.0.1.0| | | | | | | | | |
|`CUTENSOR_OP_UNKNOWN`|1.0.1.0| | | |`HIPTENSOR_OP_UNKNOWN`|5.7.0| | | | |
|`CUTENSOR_R_16BF`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_16BF`|5.7.0| | | | |
|`CUTENSOR_R_16F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_16F`|5.7.0| | | | |
|`CUTENSOR_R_16I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_16U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_32F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_32F`|5.7.0| | | | |
|`CUTENSOR_R_32I`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_32I`|5.7.0| | | | |
|`CUTENSOR_R_32U`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_32U`|5.7.0| | | | |
|`CUTENSOR_R_4I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_4U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_64F`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_64F`|5.7.0| | | | |
|`CUTENSOR_R_64I`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_64U`|2.0.0.0| | | | | | | | | |
|`CUTENSOR_R_8I`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_8I`|5.7.0| | | | |
|`CUTENSOR_R_8U`|2.0.0.0| | | |`HIPTENSOR_COMPUTE_8U`|5.7.0| | | | |
|`CUTENSOR_STATUS_ALLOC_FAILED`| | | | |`HIPTENSOR_STATUS_ALLOC_FAILED`| | | | | |
|`CUTENSOR_STATUS_ARCH_MISMATCH`| | | | |`HIPTENSOR_STATUS_ARCH_MISMATCH`| | | | | |
|`CUTENSOR_STATUS_CUBLAS_ERROR`| | | | | | | | | | |
|`CUTENSOR_STATUS_CUDA_ERROR`| | | | | | | | | | |
|`CUTENSOR_STATUS_EXECUTION_FAILED`| | | | |`HIPTENSOR_STATUS_EXECUTION_FAILED`| | | | | |
|`CUTENSOR_STATUS_INSUFFICIENT_DRIVER`| | | | |`HIPTENSOR_STATUS_INSUFFICIENT_DRIVER`| | | | | |
|`CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE`| | | | |`HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE`| | | | | |
|`CUTENSOR_STATUS_INTERNAL_ERROR`| | | | |`HIPTENSOR_STATUS_INTERNAL_ERROR`| | | | | |
|`CUTENSOR_STATUS_INVALID_VALUE`| | | | |`HIPTENSOR_STATUS_INVALID_VALUE`| | | | | |
|`CUTENSOR_STATUS_IO_ERROR`| | | | |`HIPTENSOR_STATUS_IO_ERROR`| | | | | |
|`CUTENSOR_STATUS_LICENSE_ERROR`| | | | | | | | | | |
|`CUTENSOR_STATUS_MAPPING_ERROR`| | | | | | | | | | |
|`CUTENSOR_STATUS_NOT_INITIALIZED`| | | | |`HIPTENSOR_STATUS_NOT_INITIALIZED`| | | | | |
|`CUTENSOR_STATUS_NOT_SUPPORTED`| | | | |`HIPTENSOR_STATUS_NOT_SUPPORTED`| | | | | |
|`CUTENSOR_STATUS_SUCCESS`| | | | |`HIPTENSOR_STATUS_SUCCESS`| | | | | |
|`cutensorDataType_t`|2.0.0.0| | | |`hiptensorComputeType_t`|5.7.0| | | | |
|`cutensorOperator_t`|1.0.1.0| | | |`hiptensorOperator_t`|5.7.0| | | | |
|`cutensorStatus_t`| | | | |`hiptensorStatus_t`| | | | | |

## **2. CUTENSOR Function Reference**

Unsupported


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental