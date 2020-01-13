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
#define Ndim 16
#define DataRange 50
#define PADDIM 64

void fft1d(int dimx)
{

	int N[1];
	N[0] = dimx;
	int LENGTH = N[0] ;
	cufftComplex *input = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	int i;
	for (i = 0; i < LENGTH; i++) {
		input[i].x = (i % DataRange) /100.0;
		input[i].y = 0;
	}

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
		printf("%d ",res);
	}
	size_t workSize;
	cufftGetSize1d(plan, N[0], CUFFT_C2C, 1, &workSize);
	printf("worksize : %ld and complex size %ld \n", workSize, LENGTH * sizeof(cufftComplex));

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
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

//OK
void testfp16fft(int dimx)
{

	int LENGTH= dimx;
	long long sig_size = LENGTH;

	cufftComplex *bn=(cufftComplex*)malloc(sizeof(cufftComplex)*LENGTH);
	ftype *input=(ftype*)malloc(sizeof(ftype)*LENGTH);

	ftype *acpu_data=(ftype*)malloc(sizeof(ftype)*LENGTH);
	ftype *bcpu_data=(ftype*)malloc(sizeof(ftype)*LENGTH);

	int i;
	for (i = 0; i < LENGTH; i++) {
		float temp;
		temp=i;
		acpu_data[i].x = temp;
		acpu_data[i].y = temp-temp;
	}

	for(int i=0;i<10;i++)
		printf("%f %f\n",(float)acpu_data[i].x,(float)acpu_data[i].y);

	ftype *d_adata;
	ftype *d_bdata;
	cudaMalloc(&d_adata, sizeof(ftype) * sig_size);
	cudaMalloc(&d_bdata, sizeof(ftype) * sig_size);


	cudaMemset(d_adata, 0, sig_size * sizeof(ftype));
	cudaMemset(d_bdata, 0, sig_size * sizeof(ftype));

	cudaMemcpy(d_adata, acpu_data, LENGTH * sizeof(ftype),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bdata, bcpu_data, LENGTH * sizeof(ftype),cudaMemcpyHostToDevice);




	ftype *h_idata = (ftype *) malloc(sig_size * sizeof(ftype));

	cufftHandle plan;
	cufftResult r;
	r = cufftCreate(&plan);
	size_t ws = 0;
	r = cufftXtMakePlanMany(plan, 1, &sig_size, NULL, 1, 1, CUDA_C_16F, NULL, 1,
			1, CUDA_C_16F, 1, &ws, CUDA_C_16F);

/*	assert(r == CUFFT_SUCCESS);
	r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD); // warm-up
	assert(r == CUFFT_SUCCESS);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/


	cufftXtExec(plan, d_adata, d_adata, CUFFT_FORWARD);
	cufftXtExec(plan, d_bdata, d_bdata, CUFFT_FORWARD);
/*	assert(r == CUFFT_SUCCESS);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	printf("forward FFT time for %ld samples: %fms\n", sig_size, et);*/

	cudaMemcpy(input, d_adata, LENGTH * sizeof(ftype),cudaMemcpyDeviceToHost);



}
int main(){

	fft1d(Ndim);
	testfp16fft(Ndim);

	return 0;
}
