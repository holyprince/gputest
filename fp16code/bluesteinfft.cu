#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <assert.h>

typedef half2 ftype;

#ifndef PI
#define PI 3.14159265358979323846
#endif
#define Ndim 723
#define DataRange 500
#define PADDIM 4096


void initdata(cufftComplex *input,int LENGTH)
{
	int i;
	for (i = 0; i < LENGTH; i++) {
		input[i].x = (i % DataRange);
		input[i].y = 0;
	}

}

void fft1d(int dimx)
{

	int N[1];
	N[0] = dimx;
	int LENGTH = N[0] ;
	cufftComplex *input = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	initdata(input,LENGTH);

	cufftComplex *d_inputData, *d_outData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftComplex));
	cudaMalloc((void**) &d_outData, LENGTH * sizeof(cufftComplex));

	cudaMemcpy(d_inputData, input, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cufftHandle plan;
	cufftPlan1d(&plan, N[0], CUFFT_C2C,1);
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	for (int i = 0; i < 1; i++) {

		int res=cufftExecC2C(plan, d_inputData, d_outData, CUFFT_FORWARD);

	}
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf("time is : %f \n",msecTotal1);


	size_t workSize;
	cufftGetSize1d(plan, N[0], CUFFT_C2C, 1, &workSize);
	printf("worksize : %ld and complex size %ld \n", workSize, LENGTH * sizeof(cufftComplex));



	cudaMemcpy(output, d_outData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);


	printf("=======fft\n");
	for(int i=0;i<10;i++)
		printf("%f %f\n",output[i].x,output[i].y);

	printf("Time is %f \n",msecTotal1);
	cufftDestroy(plan);
	free(input);
	free(output);
	cudaFree(d_inputData);
	cudaFree(d_outData);

}

__global__ void vectorMulti(cufftComplex *A, cufftComplex *B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
    	cufftComplex temp;
    	temp.x = A[i].x*B[i].x - A[i].y*B[i].y;
    	temp.y = A[i].x*B[i].y + A[i].y*B[i].x;
    	A[i].x=temp.x; A[i].y=temp.y;
    	//A[i]=cuCmulf(A[i],B[i]);
    }

}
__global__ void vectorMultianddiv(cufftComplex *A, cufftComplex *B, int numElements,int dimx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
    	cufftComplex temp;
    	temp.x = A[i].x*B[i].x - A[i].y*B[i].y;
    	temp.y = A[i].x*B[i].y + A[i].y*B[i].x;
    	A[i].x=temp.x/dimx; A[i].y=temp.y/dimx;

    	//A[i]=cuCmulf(A[i],B[i]);
    }

}

__global__ void vectorMulti_fp16(half2 *A, half2 *B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
       	cufftComplex temp;
        temp.x = A[i].x*B[i].x - A[i].y*B[i].y;
        temp.y = A[i].x*B[i].y + A[i].y*B[i].x;
        A[i].x=temp.x; A[i].y=temp.y;
    	//A[i]=__hmul2 (A[i],B[i]);
    }

}

__global__ void vectordivide(cufftComplex *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
       	A[i].x=A[i].x/numElements; A[i].y=A[i].y/numElements;

    }

}

__global__ void vectordivide_fp16(half2 *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
       	A[i].x= (A[i].x/(half)numElements);
     	A[i].y= (A[i].y/(half)numElements);
    }

}


__global__ void vectorMultibkstar(cufftComplex *A, int numElements,int divdata,int flag)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {

    	int index=i+divdata;

    	cufftComplex temp;
    	temp.x=cos((PI*i*i)/divdata);
    	temp.y=-sin((PI*i*i)/divdata)*flag;

       	cufftComplex temp2;
        temp2.x = A[index].x*temp.x - A[index].y*temp.y;
        temp2.y = A[index].x*temp.y + A[index].y*temp.x;
       	A[index].x=temp2.x; A[index].y=temp2.y;

    	//A[i]=cuCmulf(temp,A[i]);
    }

}

__global__ void vectorMultibkstar2(cufftComplex *A, int numElements,int divdata)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {

    	int index=i+divdata;

    	cufftComplex temp;
    	temp.x=cos((PI*i*i)/divdata);
    	temp.y=sin((PI*i*i)/divdata);

       	cufftComplex temp2;
        temp2.x = A[index].x*temp.x - A[index].y*temp.y;
        temp2.y = A[index].x*temp.y + A[index].y*temp.x;
       	A[index].x=temp2.x; A[index].y=temp2.y;

    	//A[i]=cuCmulf(temp,A[i]);
    }

}
__global__ void vectorMultibkstar_fp16(half2 *A, int numElements,int divdata)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {

    	int index=i+divdata;

    	half2 temp;
    	temp.x=cos((PI*i*i)/divdata);
    	temp.y=-sin((PI*i*i)/divdata);
 //   	A[index]=__hmul2(A[index],temp);

     	half2 temp2;
        temp2.x = A[index].x*(half)temp.x - A[index].y*temp.y;
        temp2.y = A[index].x*(half)temp.y + A[index].y*temp.x;
       	A[index].x=temp2.x; A[index].y=temp2.y;
    	//A[i]=cuCmulf(temp,A[i]);
    }

}



void bluestein(int dimx)
{
	//step1:
	int padx=PADDIM;
	int LENGTH= dimx;
	cufftComplex *an=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH);
	cufftComplex *bn=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH*2);
	cufftComplex *input=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH);

	initdata(input,LENGTH);
	for(int i=0;i<LENGTH;i++)
	{
		cufftComplex temp;
		temp.x=cos((PI*i*i)/LENGTH);
		temp.y=sin((PI*i*i)/LENGTH);
		an[i].x=input[i].x*temp.x - input[i].y*temp.y ;
		an[i].y=-(input[i].x*temp.y + input[i].y*temp.x) ;
	}
	for(int i=0;i<LENGTH*2;i++)
	{
		bn[i].x=cos((PI*(i-LENGTH)*(i-LENGTH))/LENGTH);
		bn[i].y=sin((PI*(i-LENGTH)*(i-LENGTH))/LENGTH);
	}


	//convlution
	cufftComplex *d_aData,*d_bData;

	cufftHandle plan;
	cufftPlan1d(&plan, padx, CUFFT_C2C,1);
	cudaMalloc((void**) &d_aData, padx * sizeof(cufftComplex));
	cudaMalloc((void**) &d_bData, padx * sizeof(cufftComplex));


	cudaMemset(d_aData, 0, padx * sizeof(cufftComplex));
	cudaMemset(d_bData, 0, padx * sizeof(cufftComplex));


	cudaMemcpy(d_aData, an, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bData, bn, LENGTH*2 * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cufftExecC2C(plan, d_aData, d_aData, CUFFT_FORWARD);


	cufftExecC2C(plan, d_bData, d_bData, CUFFT_FORWARD);

    int threadsPerBlock = 256;
    int blocksPerGrid =(padx + threadsPerBlock - 1) / threadsPerBlock;
    vectorMulti<<<blocksPerGrid, threadsPerBlock>>>(d_aData, d_bData, padx);

	cudaMemcpy(input, d_aData, LENGTH * sizeof(cufftComplex),cudaMemcpyDeviceToHost);


    //cudaDeviceSynchronize();
	cufftExecC2C(plan, d_aData, d_aData, CUFFT_INVERSE);

	vectordivide<<<blocksPerGrid, threadsPerBlock>>>(d_aData, padx);
	vectorMultibkstar<<<blocksPerGrid, threadsPerBlock>>>(d_aData, padx,dimx,1);


	cudaMemcpy(input, d_aData+ dimx, LENGTH * sizeof(cufftComplex),cudaMemcpyDeviceToHost);

	for(int i=0;i<10;i++)
		printf("%f %f\n",input[i].x,input[i].y);

/*
 * inverse test
	for(int i=0;i<LENGTH;i++)
	{
		cufftComplex temp;
		temp.x=cos((PI*i*i)/LENGTH);
		temp.y=sin((PI*i*i)/LENGTH);
		an[i].x=input[i].x*temp.x - input[i].y*temp.y ;
		an[i].y=input[i].x*temp.y + input[i].y*temp.x ;
	}


	for(int i=0;i<LENGTH*2;i++)
	{
		bn[i].x=cos((PI*(i-LENGTH)*(i-LENGTH))/LENGTH);
		bn[i].y=-sin((PI*(i-LENGTH)*(i-LENGTH))/LENGTH);
	}
	cudaMemset(d_aData, 0, padx * sizeof(cufftComplex));
	cudaMemset(d_bData, 0, padx * sizeof(cufftComplex));

	cudaMemcpy(d_aData, an, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bData, bn, LENGTH*2 * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cufftExecC2C(plan, d_aData, d_aData, CUFFT_FORWARD);


	cudaMemcpy(input, d_aData , LENGTH * sizeof(cufftComplex),cudaMemcpyDeviceToHost);



	cufftExecC2C(plan, d_bData, d_bData, CUFFT_FORWARD);

    threadsPerBlock = 256;
    blocksPerGrid =(padx + threadsPerBlock - 1) / threadsPerBlock;
    vectorMultianddiv<<<blocksPerGrid, threadsPerBlock>>>(d_aData, d_bData, padx,dimx);

    //cudaDeviceSynchronize();
	cufftExecC2C(plan, d_aData, d_aData, CUFFT_INVERSE);
	vectordivide<<<blocksPerGrid, threadsPerBlock>>>(d_aData, padx);

	vectorMultibkstar<<<blocksPerGrid, threadsPerBlock>>>(d_aData, padx,dimx,-1);

	cudaMemcpy(input, d_aData + dimx, LENGTH * sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	printf("\n");
	for(int i=0;i<10;i++)
		printf("%f %f\n",input[i].x,input[i].y);
*/

}

void bluesteininverse(int dimx)
{
	int padx=PADDIM;
	int LENGTH= dimx;
	cufftComplex *an=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH);
	cufftComplex *bn=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH*2);
	cufftComplex *input=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH);

	initdata(input,LENGTH);


	for(int i=0;i<LENGTH;i++)
	{
		cufftComplex temp;
		temp.x=cos((PI*i*i)/LENGTH);
		temp.y=sin((PI*i*i)/LENGTH);
		an[i].x=input[i].x*temp.x - input[i].y*temp.y ;
		an[i].y=input[i].x*temp.y + input[i].y*temp.x ;
	}

	for(int i=0;i<LENGTH*2;i++)
	{
		bn[i].x=cos((PI*(i-LENGTH)*(i-LENGTH))/LENGTH);
		bn[i].y=-sin((PI*(i-LENGTH)*(i-LENGTH))/LENGTH);
	}


	//convlution
	cufftComplex *d_aData,*d_bData;

	cufftHandle plan;
	cufftPlan1d(&plan, padx, CUFFT_C2C,1);
	cudaMalloc((void**) &d_aData, padx * sizeof(cufftComplex));
	cudaMalloc((void**) &d_bData, padx * sizeof(cufftComplex));


	cudaMemset(d_aData, 0, padx * sizeof(cufftComplex));
	cudaMemset(d_bData, 0, padx * sizeof(cufftComplex));


	cudaMemcpy(d_aData, an, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bData, bn, LENGTH*2 * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	cufftExecC2C(plan, d_aData, d_aData, CUFFT_FORWARD);


	cufftExecC2C(plan, d_bData, d_bData, CUFFT_FORWARD);

    int threadsPerBlock = 256;
    int blocksPerGrid =(padx + threadsPerBlock - 1) / threadsPerBlock;
    vectorMultianddiv<<<blocksPerGrid, threadsPerBlock>>>(d_aData, d_bData, padx,dimx);



	cufftExecC2C(plan, d_aData, d_aData, CUFFT_INVERSE);
	vectordivide<<<blocksPerGrid, threadsPerBlock>>>(d_aData, padx);

	vectorMultibkstar<<<blocksPerGrid, threadsPerBlock>>>(d_aData, padx,dimx,-1);

	cudaMemcpy(input, d_aData + dimx, LENGTH * sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	printf("\n");
	for(int i=0;i<10;i++)
		printf("%f %f\n",input[i].x,input[i].y);

}






int main(){

	fft1d(Ndim);
	bluestein(Ndim);
	//fft1d(4096);
	printf("FP16: \n");




	return 0;
}


