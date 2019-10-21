#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"



//#define PRINT

#define REP_TIMES 100

void testmoduleGPU(int dimx,int dimy,int dimz) {
	int N[3];
	N[0] = dimx; N[1] = dimy; N[2] = dimz;
	int LENGTH = N[0] * N[1] * N[2];
	cufftComplex *input = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*) malloc(
			LENGTH * sizeof(cufftComplex));
	int i;
	for (i = 0; i < N[0] * N[1]; i++) {
		input[i].x = i % 1000;
		input[i].y = 0;
	}

	cufftComplex *d_inputData, *d_outData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftComplex));
	cudaMalloc((void**) &d_outData, LENGTH * sizeof(cufftComplex));

	cudaMemcpy(d_inputData, input, LENGTH * sizeof(cufftComplex),
			cudaMemcpyHostToDevice);

	cufftHandle plan;

	int t= cufftPlan3d(&plan, N[0], N[1], N[2], CUFFT_C2C);


	size_t* worksize;
	worksize = (size_t*) malloc(sizeof(size_t) * 2);
	cufftGetSize3d(plan,N[0], N[1], N[2],CUFFT_C2C, worksize);
	printf("make : %d %ld \n",t,worksize[0]);
	cufftExecC2C(plan, d_inputData, d_outData, CUFFT_INVERSE);

	cudaMemcpy(output, d_outData, LENGTH * sizeof(cufftComplex),
			cudaMemcpyDeviceToHost);

	for (i = 0; i < 10; i++) {
		printf("%f %f \n", output[i].x, output[i].y);
	}

	cufftDestroy(plan);
	free(input);
	free(output);
	cudaFree(d_inputData);
	cudaFree(d_outData);
}

int main() {
	testmoduleGPU(200,200,200);
}
