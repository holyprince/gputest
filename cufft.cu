#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"



//#define PRINT

#define REP_TIMES 100

float testmoduleGPU(int dimx,int dimy) {
	int N[2];
	N[0] = dimx, N[1] = dimy;
	int LENGTH = N[0] * N[1];
	cufftComplex *input = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*) malloc(
			LENGTH * sizeof(cufftComplex));
	int i;
	for (i = 0; i < N[0] * N[1]; i++) {
		input[i].x = i % 1000;
		input[i].y = 0;
	}

	cufftComplex *d_inputData, *d_outData;
	cudaMalloc((void**) &d_inputData, N[0] * N[1] * sizeof(cufftComplex));
	cudaMalloc((void**) &d_outData, N[0] * N[1] * sizeof(cufftComplex));

	cudaMemcpy(d_inputData, input, N[0] * N[1] * sizeof(cufftComplex),
			cudaMemcpyHostToDevice);

	cufftHandle plan;

	cufftPlan2d(&plan, N[0], N[1], CUFFT_C2C);

	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
	for (int i = 0; i < 100; i++) {
		//cufftExecC2C(plan, d_inputData, d_outData, CUFFT_FORWARD);
		cufftExecC2C(plan, d_inputData, d_outData, CUFFT_INVERSE);
	}

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	cudaMemcpy(output, d_outData, LENGTH * sizeof(cufftComplex),
			cudaMemcpyDeviceToHost);
/*
	for (i = 0; i < 10; i++) {
		printf("%f %f \n", output[i].x, output[i].y);
	}
	*/
	cufftDestroy(plan);
	free(input);
	free(output);
	cudaFree(d_inputData);
	cudaFree(d_outData);
	return msecTotal1;
}

int main() {
	double timeres[10];
    //128=2^7    ; 8192=2^13
	int pownum=3;
	for(pownum=7;pownum<=13;pownum++)
	{
		double avertime = 0;
		for (int i = 0; i < 10; i++) {
			timeres[i] = testmoduleGPU(pow(2,pownum),pow(2,pownum));
			printf("ITER %f ", timeres[i]);
			avertime += timeres[i];
		}
		printf("\n AVER %f \n", avertime / 10);
	}
}
