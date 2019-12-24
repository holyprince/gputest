### 10.21
1.计算C2R与R2C
cufftMakePlan1d 与batch=3的1维FFT效果相同
使用带stride的1维fft可以进行拆分，但是XY顺序对结果有影响
2.layout排布
高维R2C的结果好像不能直接使用共轭矩阵变换

### 10月23日
1.二维的实数到复数测试没有问题(hermitian) [layoutdim2.cu]
2.三维的实数到复数的复制也没有问题 [layoutdim3.cu]


### 12月24日
1.FFT内存估算：cufftGetSize2d 比 cufftEstimate2d 更加精确，使用前者。