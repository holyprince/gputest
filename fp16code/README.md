# FP16 FFT
1. compile arg : -gencode arch=compute_70,code=sm_70 
2. fp16ffttest.cu
3. error: calling a __device__ function("operator=") from a __host__ function("") is not allowed
   不能直接将int或者整数赋值给half类型
   
4. fp16 __half2 arithmetic ?
__hmul2 这个操作是直接像素相乘，并不是复数乘法

https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html#group__CUDA__MATH__INTRINSIC__HALF
