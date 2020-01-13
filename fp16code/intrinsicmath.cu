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



void testarithmetic()
{
	ftype *d_adata;
	ftype *acpu_data=(ftype* )malloc(sizeof(ftype) * 2);
	cudaMalloc(&d_adata, sizeof(ftype) * 2);
	cudaMemset(d_adata, 0, 2 * sizeof(ftype));


    int threadsPerBlock = 256;
    int blocksPerGrid =(2 + threadsPerBlock - 1) / threadsPerBlock;
	vectorMulti_multitest<<<blocksPerGrid, threadsPerBlock>>>(d_adata, 2);
	cudaMemcpy(acpu_data, d_adata, 2 * sizeof(ftype),cudaMemcpyDeviceToHost);
	printf("%f %f \n",(float)acpu_data[0].x,(float)acpu_data[0].y);
}
int main(){

	testarithmetic();
	return 0;
}