#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include <cuda_runtime_api.h>
#include "cuda.h"
#define STREAMNUM 2
void init()
{

}
__global__ void MyKernel(float *a,float* b,int nx)
{

}

//#define PINDIRECT
int main() {

	int size=sizeof(float)*100;
	cudaStream_t stream[2];
	for (int i = 0; i < 2; ++i)
		cudaStreamCreate(&stream[i]);
	float* hostPtr;
	float* inputDevPtr,*outputDevPtr;
#ifdef PINDIRECT
    cudaMallocHost(&hostPtr, 2 * size);
#else
    hostPtr = (float*)malloc(2 * size);
    cudaHostRegister(hostPtr,2 * size,CU_MEMHOSTALLOC_DEVICEMAP);
#endif
    cudaMalloc((void**)&inputDevPtr,size*STREAMNUM);
    cudaMalloc((void**)&outputDevPtr,size*STREAMNUM);

    init();

	for (int i = 0; i < 2; i++) {

		cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size,cudaMemcpyHostToDevice, stream[i]);

		MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size,inputDevPtr + i * size, size);

		cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size,cudaMemcpyDeviceToHost, stream[i]);
	}

	for (int i = 0; i < 2; i++) {
		cudaStreamSynchronize(stream[i]);
		cudaStreamDestroy(stream[i]);
		//CPU code can execute cpu and gpu the same time ;
	}
}
