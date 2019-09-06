nvcc -I/GPUFS/ict_zyliu_2/code/inc bandwidthTest.cu

需要解压一个include压缩包
文件来自 cuda sample

带宽测量
两种内存：可分页内存与页锁定内存
三种传输方式：固定大小(Quick)、范围大小(range)、递增大小(smooth)


