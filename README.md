# gpu test
Some gpu code

Error handle 
https://blog.csdn.net/monroed/article/details/70307312

cuda Init 
https://blog.csdn.net/fengtian12345/article/details/80549410

trick internet
https://developer.nvidia.com/cuda-education

multi gpu test code from 
https://github.com/bblakeley/Multi_GPU_FFT_check/blob/master/Multi_GPU_FFT_check.cu
All single GPU cuFFT FFTs return output the data in natural order, that is the ordering of the result is the same as if a DFT had been performed on the data. Some Fast Fourier Transforms produce intermediate results where the data is left in a permutation of the natural output. When batch is one, data is left in the GPU memory in a permutation of the natural output. [cufftlibrary.pdf]

typedef enum cufftResult_t {
CUFFT_SUCCESS = 0, // The cuFFT operation was successful
CUFFT_INVALID_PLAN = 1, // cuFFT was passed an invalid plan handle
CUFFT_ALLOC_FAILED = 2, // cuFFT failed to allocate GPU or CPU memory
CUFFT_INVALID_TYPE = 3, // No longer used
CUFFT_INVALID_VALUE = 4, // User specified an invalid pointer or
parameter
CUFFT_INTERNAL_ERROR = 5, // Driver or internal cuFFT library error
CUFFT_EXEC_FAILED = 6, // Failed to execute an FFT on the GPU
CUFFT_SETUP_FAILED = 7, // The cuFFT library failed to initialize
CUFFT_INVALID_SIZE = 8, // User specified an invalid transform size
CUFFT_UNALIGNED_DATA = 9, // No longer used
CUFFT_INCOMPLETE_PARAMETER_LIST = 10, // Missing parameters in call
CUFFT_INVALID_DEVICE = 11, // Execution of a plan was on different GPU than
plan creation
CUFFT_PARSE_ERROR = 12, // Internal plan database error
CUFFT_NO_WORKSPACE = 13 // No workspace has been provided prior to plan
execution
CUFFT_NOT_IMPLEMENTED = 14, // Function does not implement functionality for
parameters given.
CUFFT_LICENSE_ERROR = 15, // Used in previous versions.
CUFFT_NOT_SUPPORTED = 16 // Operation is not supported for parameters given.
} cufftResult;