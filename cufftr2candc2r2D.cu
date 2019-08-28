

#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"



int main() {

	int N[2];
	N[0]=3;N[1]=3;

	cufftReal *d_real_data,*c_real_data;
	c_real_data = (cufftReal*) malloc(sizeof(cufftReal)*N[0]*N[1]);
    cudaMalloc((void**)&d_real_data, N[0]*N[1] * sizeof(cufftReal));
	c_real_data[0]=0;
	c_real_data[1]=2;
	c_real_data[2]=4;
	c_real_data[3]=6;
	c_real_data[4]=1;
	c_real_data[5]=3;
	c_real_data[6]=5;
	c_real_data[7]=7;
	c_real_data[8]=4;

	cudaMemcpy(d_real_data,c_real_data,sizeof(cufftReal)*N[0]*N[1],cudaMemcpyHostToDevice);

	cufftComplex *d_comp_data,*c_comp_data;
	cudaMalloc((void**)&d_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1));
	c_comp_data = (cufftComplex*) malloc(sizeof(cufftComplex)*N[0]*(N[1]/2+1));

	cufftHandle cufftForwrdHandle;
	cufftPlan2d(&cufftForwrdHandle, N[0], N[1] , CUFFT_R2C);
	cufftExecR2C(cufftForwrdHandle, d_real_data, d_comp_data);
	cudaDeviceSynchronize();
	cudaMemcpy(c_comp_data,d_comp_data,sizeof(cufftComplex)*N[0]*(N[1]/2+1),cudaMemcpyDeviceToHost);
	cufftDestroy(cufftForwrdHandle);

	for(int i=0;i<N[0]*(N[1]/2+1);i++)
		printf("%f %f \n",c_comp_data[i].x,c_comp_data[i].y);

// C2R

	cufftHandle cufftInverseHandle;
	cudaMemset(d_real_data,0,N[0]*N[1]*sizeof(cufftReal));
	cufftPlan2d(&cufftInverseHandle, N[0],N[1], CUFFT_C2R);
	cufftExecC2R(cufftInverseHandle, d_comp_data,d_real_data);
	cudaMemcpy(c_real_data,d_real_data,sizeof(cufftReal)*N[0]*N[1],cudaMemcpyDeviceToHost);
	for(int i=0;i<N[0]*N[1];i++)
		printf("%f ",c_real_data[i]);


}
/*
ref: https://blog.csdn.net/congwulong/article/details/7576012
[0,2,4]
[6,1,3]
[5,7,4]

32 0.5+0.86i 0.5-0.86i;
-7+5.2i -1-1.73i -8.5-6.06i;
-7-5.2i -8.5+6.06i -1+1.73i;

32.000000 0.000000
0.500000 0.866025
-7.000000 5.196152
-1.000000 -1.732050
-7.000000 -5.196152
-8.500000 6.062178

inverse change :
0.000000 18.000000 36.000000 54.000000 9.000003 27.000000 45.000000 63.000000 36.000000

 */


