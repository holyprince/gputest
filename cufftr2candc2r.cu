
#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"






#define NX 256
#define NY 128
#define NRANK 2
#define BATCH 1



int main() {

	int N=4;
	cufftReal *d_real_data,*c_real_data;
	c_real_data = (cufftReal*) malloc(sizeof(cufftReal)*4);
    cudaMalloc((void**)&d_real_data, N * sizeof(cufftReal));
	c_real_data[0]=1;
	c_real_data[1]=2;
	c_real_data[2]=3;
	c_real_data[3]=4;

	cudaMemcpy(d_real_data,c_real_data,sizeof(cufftReal)*4,cudaMemcpyHostToDevice);

	cufftComplex *d_comp_data,*c_comp_data;
	cudaMalloc((void**)&d_comp_data, sizeof(cufftComplex)*(N/2+1));
	c_comp_data = (cufftComplex*) malloc(sizeof(cufftComplex)*(N/2+1));

	cufftHandle cufftForwrdHandle;
	cufftPlan1d(&cufftForwrdHandle, N, CUFFT_R2C, 1);
	cufftExecR2C(cufftForwrdHandle, d_real_data, d_comp_data);
	cudaDeviceSynchronize();
	cudaMemcpy(c_comp_data,d_comp_data,sizeof(cufftComplex)*(N/2+1),cudaMemcpyDeviceToHost);
	cufftDestroy(cufftForwrdHandle);

	for(int i=0;i<(N/2+1);i++)
		printf("%f %f \n",c_comp_data[i].x,c_comp_data[i].y);

// C2R

	cufftHandle cufftInverseHandle;
	cudaMemset(d_real_data,0,N*sizeof(cufftReal));
	cufftPlan1d(&cufftInverseHandle, N, CUFFT_C2R, 1);
	cufftExecC2R(cufftInverseHandle, d_comp_data,d_real_data);
	cudaMemcpy(c_real_data,d_real_data,sizeof(cufftReal)*N,cudaMemcpyDeviceToHost);
	for(int i=0;i<N;i++)
		printf("%f ",c_real_data[i]);


}
/*
ref: https://blog.csdn.net/congwulong/article/details/7576012
10.000000 -0.000000
-2.000000 2.000000
-2.000000 0.000000
4.000001 7.999999 11.999999 16.000000
 */