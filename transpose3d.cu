#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include "timer.h"

#define NX 512
#define NY 512
#define NZ 512

#define NXY NX*NY
#define K 32

void initdata(float data[])
{
	for(int zi=0;zi< NZ ;zi++)
		for(int yi=0; yi< NY;yi++)
			for(int xi=0; xi< NX;xi++)
			{
				data[zi*NXY+ yi*NX + xi] = zi*NXY+ yi*NX + xi;
			}

}
void printdata(float data[])
{

	for(int zi=0;zi< NZ ;zi++){

		for(int yi=0; yi< NY;yi++) {
			for(int xi=0; xi< NX;xi++)
			{
				printf("%.2f ",data[zi*NXY+ yi*NX + xi]);
			}
			printf("\n");
		}
		printf("===========\n");
	}



}

void transpose_CPU(float in[], float out[]) {

	for(int zi=0;zi< NZ ;zi++)
		for(int yi=0; yi< NY;yi++)
			for(int xi=0; xi< NX;xi++)
				out[yi*NXY+ zi*NX + xi] = in[zi*NXY+ yi*NX + xi];
}


__global__ void transpose(float in[], float out[])
{
    unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int k = blockIdx.z * blockDim.z;
    //unsigned int idx = i * N + j;
    if(i < NX && j < NY && k<NZ/2 )
    {

    	out[j * NXY + k *NX+ i] = in[k * NXY + j*NX+ i];
    	//out[k * NXY + j *NX+ i] = in[k * NXY + j*NX+ i];
    }
}


__global__ void transpose_parallel_per_element_tiled(float in[], float out[])
{
  int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

  int x = threadIdx.x, y = threadIdx.y;

  __shared__ float tile[K][K];

  //tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * N];
  __syncthreads();
  //out[(out_corner_i + x) + (out_corner_j + y) * N] = tile[x][y];
}

void testbandwidth()
{
	int devicenum=0;
    cudaSetDevice(devicenum);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devicenum);
    printf("  Memory Clock rate:    %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:    %d-bit\n",   deviceProp.memoryBusWidth);
    printf(" Theory peak : %.2f GB/s \n",(deviceProp.memoryBusWidth*deviceProp.memoryClockRate * 1e-3f)/8/1000.0);
}
int main(int argc, char **argv) {


	testbandwidth();


	int numbytes =NX * NY * NZ* sizeof(float);
	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	initdata(in);

	float *d_in, *d_out; // on Device
	cudaMalloc((void **)&d_in, numbytes);
	cudaMalloc((void **)&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);


	dim3 threads(K,K);
	dim3 blocks((NX + threads.x - 1) / threads.x, (NY + threads.y - 1) / threads.y, NZ/2);


	for(int i=0;i<20;i++)
	{
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
	transpose<<<blocks, threads>>>(d_in, d_out);
//	transpose_parallel_per_element_tiled<<<blocks, threads>>>(d_in, d_out);
//	cudaDeviceSynchronize();
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf("Time : %f \n",msecTotal1);
	}
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);


	//printdata(out);
	return 0;
}
