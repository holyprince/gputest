# FP16 FFT
1. compile arg : -gencode arch=compute_70,code=sm_70 
2. fp16ffttest.cu : 
3. error: calling a __device__ function("operator=") from a __host__ function("") is not allowed
   不能直接将int或者整数赋值给half类型
   
4. fp16 __half2 arithmetic ?
__hmul2 这个操作是直接像素相乘，并不是复数乘法

https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html#group__CUDA__MATH__INTRINSIC__HALF

5. 任意维度的FFT：[bluesteinfft.cu]
	使用bluestein算法+2的N次方cufft
	
6. fp16的任意维度FFT：[bluestein_fp16.cu]
	使用bluestein算法与 2的N次方FP16算法

7. 使用tensorcore的fp16算法--性能要求 []
与应用结合：

实现思路一：
1.将数据进行normalization，然后直接调用FP16的三维版本


实现思路二：
1.将3DFFT分开实现  [dividefft.cu]
	将3维拆分成3个一维或者一个二维和一个一维
		结果不太一致[可能是在误差范围以内]
	第三种拆分方法：先做一个一维xdim，再做一个二维 ydim\*zdim
		这种情况yz二维整体，好像没有办法使用stride形式表示
2.将其中一个维度替换成fp16计算
	按照大小来说，第一个维度换为fp16比较合适

优点：数值范围可能可以cover住
缺点：内存额外消耗？


