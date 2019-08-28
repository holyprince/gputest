#include "cuda_runtime.h"
#include "cufft.h"   //   /usr/local/cuda-9.0/include
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void vectorAdd(int *A, int *B, int *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        //C[i].x = A[i] * B[i];
        C[i] = A[i]+B[i];
    }

    //printf("%f from gpu \n",C[i]);

}

int main()
{
	int *a,*b;
	int N=2000;
	a = (int *)malloc(sizeof(int)*N);
	b = (int *)malloc(sizeof(int)*N);
	for(int i=0;i<N;i++)
	{
		a[i]=1;
		b[i]=2;
	}
	int *c;
	c = (int *)malloc(sizeof(int)*N);
	int *d_a,*d_b,*d_c;
	cudaMalloc((void**) &d_a, N * sizeof(int));
	cudaMalloc((void**) &d_b, N * sizeof(int));
	cudaMemcpy(d_a, a, N * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_c, N * sizeof(int));
    int threadsPerBlock = 256;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
	cudaMemcpy(c, d_c, N * sizeof(int),cudaMemcpyDeviceToHost);    
	for(int i=0;i<10;i++)
		printf("%d \n",c[i]);
}