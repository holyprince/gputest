
// System includes
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

//CUFFT Header file
#include <cufftXt.h>
#include "timer.h"

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

// Complex data type
typedef float2 Complex;


#define N 512



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

	Complex *f = (Complex*) malloc(sizeof(Complex) * N * N *N);
	float *u_a = (float*) malloc(sizeof(float) * N * N*N);
	Complex *h_d_out = (Complex *) malloc(sizeof(Complex) * N * N*N);

	for (int i = 0; i < N * N*N; i++) {
		f[i].x = i % 5000 ;
		f[i].y = 0;
	}
	StartTimer();
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
	result = cufftMakePlan3d(planComplex, N, N,N, CUFFT_C2C, worksize);

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


	// cufftXtMemcpy() - Copy data from multiple GPUs to host
	result = cufftXtMemcpy(planComplex, h_d_out, d_d_f, CUFFT_COPY_DEVICE_TO_HOST);

	if (result != CUFFT_SUCCESS) {
		printf("*XtMemcpy failed\n");
		exit(EXIT_FAILURE);
	}


	// cleanup memory

	free(h_d_out);
	free(worksize);
    printf("  GPU Processing time: %f (ms)\n\n", GetTimer());

	return 0;
}
