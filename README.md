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