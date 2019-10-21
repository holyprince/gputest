### 10.21
1.计算C2R与R2C
cufftMakePlan1d 与batch=3的1维FFT效果相同
使用带stride的1维fft可以进行拆分，但是XY顺序对结果有影响
2.layout排布
高维R2C的结果好像不能直接使用共轭矩阵变换