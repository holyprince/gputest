#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include "timer.h"

#define NX 2048
#define NY 2048

#define K 16

void initdata(float data[])
{
		for(int yi=0; yi< NY;yi++)
			for(int xi=0; xi< NX;xi++)
			{
				data[yi*NX + xi] = yi*NX + xi;
			}

}
void printdata(float data[])
{



		for(int yi=0; yi< NY;yi++) {
			for(int xi=0; xi< NX;xi++)
			{
				printf("%.2f ",data[yi*NX + xi]);
			}
			printf("\n");
		}
}
void transpose_CPU(float in[], float out[]) {


		for(int yi=0; yi< NY;yi++)
			for(int xi=0; xi< NX;xi++)
				out[ xi*NX + yi] = in[ yi*NX + xi];
}

__global__ void transpose(float in[], float out[])
{
    unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned int k = blockIdx.z * blockDim.z;
    //unsigned int idx = i * N + j;
    if(i < NX && j < NY )
    {

    	out[ i *NX+ j] = in[j*NX+ i];
    	//out[k * NXY + j *NX+ i] = in[k * NXY + j*NX+ i];
    }
}


__global__ void transpose_parallel_per_element_tiled1(float in[], float out[])
{
  int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

  int x = threadIdx.x, y = threadIdx.y;

  __shared__ float tile[K][K];

  tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * NX];
  __syncthreads();
  out[(out_corner_i + x) + (out_corner_j + y) * NX] = tile[x][y];
}


__global__ void transpose_parallel_per_element_tiled2(float in[], float out[])
{
  int in_corner_i = blockIdx.x * K, in_corner_j = blockIdx.y * K;
  int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;

  int x = threadIdx.x, y = threadIdx.y;

  __shared__ float tile[K][K+1];

  tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y) * NX];
  __syncthreads();
  out[(out_corner_i + x) + (out_corner_j + y) * NX] = tile[x][y];
}
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_cudaapi(float in[], float out[],int width,int height)
{
	int xIndex = blockIdx.x *TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y *TILE_DIM + threadIdx.y;

	int index_in = xIndex+ width*yIndex;
	int index_out = yIndex+ height *xIndex;
	for(int i=0;i<TILE_DIM ; i+=BLOCK_ROWS){
		out[index_out +i]= in[index_in + i* width];
	}
}

__global__ void transpose_cudaapi_shared(float in[], float out[],int width,int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int xIndex = blockIdx.x *TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y *TILE_DIM + threadIdx.y;
	int index_in = xIndex+ width*yIndex;

	xIndex= blockIdx.y * TILE_DIM+ threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex+ yIndex*height;

	for(int i=0;i<TILE_DIM ; i+=BLOCK_ROWS){
		tile[threadIdx.y+i][threadIdx.x] =  in[index_in + i* width];
	}
	__syncthreads();

	for(int i=0;i<TILE_DIM ; i+=BLOCK_ROWS ){
		out[index_out + i*height] = tile[threadIdx.x][threadIdx.y+i];
	}
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


	int numbytes =NX * NY * sizeof(float);
	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	initdata(in);

//	transpose_CPU(in,out);

	float *d_in, *d_out; // on Device
	cudaMalloc((void **)&d_in, numbytes);
	cudaMalloc((void **)&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);


	dim3 threads(K,K);
	dim3 blocks((NX + threads.x - 1) / threads.x, (NY + threads.y - 1) / threads.y);


	for(int i=0;i<21;i++)
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
	printf("Time1  : %f \n",msecTotal1);
	}


	for(int i=0;i<21;i++)
	{
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	transpose_parallel_per_element_tiled1<<<blocks, threads>>>(d_in, d_out);

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf("Time2  : %f \n",msecTotal1);
	}

	for(int i=0;i<21;i++)
	{
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	transpose_parallel_per_element_tiled2<<<blocks, threads>>>(d_in, d_out);

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf("Time3  : %f \n",msecTotal1);
	}

	dim3 threads1(TILE_DIM, BLOCK_ROWS);
	dim3 blocks1(NX / TILE_DIM, NY / TILE_DIM);
	//dim3 blocks1((NX + TILE_DIM - 1) / TILE_DIM, (NY + TILE_DIM - 1) / TILE_DIM);

	for(int i=0;i<21;i++)
	{
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	transpose_cudaapi<<<blocks1, threads1>>>(d_in, d_out,NX,NY);

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf("Time4  : %f \n",msecTotal1);
	}
	for(int i=0;i<21;i++)
	{
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	transpose_cudaapi_shared<<<blocks1, threads1>>>(d_in, d_out,NX,NY);

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf("Time5  : %f \n",msecTotal1);
	}

	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);


//	printdata(out);
	return 0;
}
