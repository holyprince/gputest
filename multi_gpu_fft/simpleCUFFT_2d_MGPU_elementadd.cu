/**
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/


// System includes
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

//CUFFT Header file
#include <cufftXt.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

// Complex data type
typedef float2 Complex;


#define N 32

// Forward Declaration
__global__ void vectorAdd(cufftComplex *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
    	A[i].y = i;
        A[i].x = A[i].x+(i)*10;
    }
}


__global__ void vectorAddself(cufftComplex *A, int numElements, int curnum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int newrow= i / (N/2);
    int newcol= i % (N/2);
    int realindex;
    if (i < numElements)
    {
    	realindex = newrow * N + newcol + (curnum *N/2);
    	A[i].y = realindex;
    	A[i].x = A[i].x + realindex*10;
    }


}
void vectorAdd2(cudaLibXtDesc *d_ft, int nGPUs)
{
    int device ;
	int threadsPerBlock = 256;
	int blocksPerGrid =(N/2*N + threadsPerBlock - 1) / threadsPerBlock;
    for(int i=0; i < nGPUs ; i++)
    {
        device = d_ft->descriptor->GPUs[i];
        cudaSetDevice(device) ;
        vectorAddself<<<blocksPerGrid,threadsPerBlock>>>((cufftComplex*) d_ft->descriptor->data[i], N/2*N, i);

    }

    // Wait for device to finish all operation
    for(int i=0; i< nGPUs ; i++)
    {
        device = d_ft->descriptor->GPUs[i];
        cudaSetDevice( device );
        cudaDeviceSynchronize();
    }

}


///////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{

	int threadsPerBlock;
	int blocksPerGrid ;
	int GPU_N;
	(cudaGetDeviceCount(&GPU_N));
	int nGPUs = 2;
	int *whichGPUs;
	whichGPUs = (int*) malloc(sizeof(int) * nGPUs);

	Complex *f = (Complex*) malloc(sizeof(Complex) * N * N);
	float *u_a = (float*) malloc(sizeof(float) * N * N);
	Complex *h_d_out = (Complex *) malloc(sizeof(Complex) * N * N);

	for (int i = 0; i < N * N; i++) {
		f[i].x = i % 5000 ;
		f[i].y = 0;
	}

	// cufftCreate() - Create an empty plan
	cufftResult result;
	cufftHandle planComplex;
	result = cufftCreate(&planComplex);
	if (result != CUFFT_SUCCESS) {
		printf("cufftCreate failed\n");
		exit(EXIT_FAILURE);
	}

	// cufftXtSetGPUs() - Define which GPUs to use
	result = cufftXtSetGPUs(planComplex, nGPUs, whichGPUs);

	if (result == CUFFT_INVALID_DEVICE) {
		printf("This sample requires two GPUs on the same board.\n");
		printf("No such board was found. Waiving sample.\n");

	} else if (result != CUFFT_SUCCESS) {
		printf("cufftXtSetGPUs failed\n");

	}

	//Print the device information to run the code
	printf("\nRunning on GPUs\n");

	size_t* worksize;
	worksize = (size_t*) malloc(sizeof(size_t) * nGPUs);

	// cufftMakePlan2d() - Create the plan
	result = cufftMakePlan2d(planComplex, N, N, CUFFT_C2C, worksize);

	if (result != CUFFT_SUCCESS) {
		printf("*MakePlan* failed\n");
		exit(EXIT_FAILURE);
	}

	// Create a variable on device
	// d_f - variable on device to store the input data
	// d_d_f - variable that store the natural order of d_f data
	// d_out - device output
	cudaLibXtDesc *d_f, *d_d_f;

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **) &d_f, CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) {
		printf("*XtMalloc failed\n");
		exit(EXIT_FAILURE);
	}
	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **) &d_d_f, CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) {
		printf("*XtMalloc failed\n");
		exit(EXIT_FAILURE);
	}

	// cufftXtMemcpy() - Copy the data from host to device
	result = cufftXtMemcpy(planComplex, d_f, f, CUFFT_COPY_HOST_TO_DEVICE);
	if (result != CUFFT_SUCCESS) {
		printf("*XtMemcpy failed\n");
		exit(EXIT_FAILURE);
	}

	// cufftXtExecDescriptorC2C() - Execute FFT on data on multiple GPUs
	printf("Forward 2d FFT on multiple GPUs\n");
	result = cufftXtExecDescriptorC2C(planComplex, d_f, d_f, CUFFT_FORWARD);
	if (result != CUFFT_SUCCESS) {
		printf("*XtExecC2C  failed\n");
		exit(EXIT_FAILURE);
	}
//in this case reorder is not necessary , so next step not using d_d_f
	result = cufftXtMemcpy(planComplex, d_d_f, d_f, CUFFT_COPY_DEVICE_TO_DEVICE);
	if (result != CUFFT_SUCCESS) {
		printf("*XtMemcpy failed\n");
		exit(EXIT_FAILURE);
	}

	vectorAdd2(d_f,nGPUs);


	// cufftXtMemcpy() - Copy data from multiple GPUs to host
	result = cufftXtMemcpy(planComplex, h_d_out, d_f, CUFFT_COPY_DEVICE_TO_HOST);

	if (result != CUFFT_SUCCESS) {
		printf("*XtMemcpy failed\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < N*N ; i++) {
		if(i%32==0)
			printf("\n");
		printf("%f %f ", h_d_out[i].x, h_d_out[i].y);
	}

	// cleanup memory

	free(h_d_out);
	free(worksize);

	result = cufftXtFree(d_f);
	if (result != CUFFT_SUCCESS) {
		printf("*XtFree failed\n");
		exit(EXIT_FAILURE);
	}
	cufftXtFree(d_d_f);
	// cufftDestroy() - Destroy FFT plan
	result = cufftDestroy(planComplex);
	if (result != CUFFT_SUCCESS) {
		printf("cufftDestroy failed: code %d\n", (int) result);
		exit(EXIT_FAILURE);
	}
	printf(" single gpu for fft \n \n \n");

	int Ndim[2];
	Ndim[0] = N, Ndim[1] = N;
	int LENGTH = Ndim[0] * Ndim[1];

	cufftComplex *output = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));

	cufftComplex *d_inputData, *d_outData;
	cudaMalloc((void**) &d_inputData, Ndim[0] * Ndim[1] * sizeof(cufftComplex));
	cudaMalloc((void**) &d_outData, Ndim[0] * Ndim[1] * sizeof(cufftComplex));

	cudaMemcpy(d_inputData, f, Ndim[0] * Ndim[1] * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	cufftHandle plan;

	cufftPlan2d(&plan, Ndim[0], Ndim[1], CUFFT_C2C);

	cufftExecC2C(plan, d_inputData, d_inputData, CUFFT_FORWARD);

	threadsPerBlock = 256;
    blocksPerGrid =(N*N + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_inputData, N*N);
	cudaMemcpy(output, d_inputData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N*N; i++) {
		if(i%32==0)
			printf("\n");
			printf("%f %f ", output[i].x, output[i].y);
	}

	cufftDestroy(plan);
	free(output);
	cudaFree(d_inputData);
	cudaFree(d_outData);
	return 0;
}



