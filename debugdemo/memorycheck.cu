#include "stdio.h"
#include "stdlib.h"
#include "fftw3.h"
#include "cufft.h"
#include "cuda_runtime.h"

int main() {

	int N[3];
	N[0]=723;N[1]=723;N[2]=723;

	int Nsum=N[0]*N[1]*N[2];
	double *c_real_data;
	c_real_data = (double*) malloc(sizeof(double)*Nsum);

	for(int i=0;i<Nsum;i++)
	{
		c_real_data[i] = i%700;
	}
	cufftResult cr;
	cufftHandle cufftForwrdHandle;
	cr =  cufftPlan3d(&cufftForwrdHandle, N[0], N[1],N[2], CUFFT_C2R);
	size_t workSize,freeMem,totalMem;
	cufftGetSize(cufftForwrdHandle, &workSize);
	printf("%d and worksize : %ld \n ",cr,workSize);
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("%ld %ld \n",freeMem,totalMem);
}