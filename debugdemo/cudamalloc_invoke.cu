#include "cuda_runtime.h"
#include "cufft.h"
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void vectorMulti(double *A, double *B, cufftComplex *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i].x = A[i] * B[i];

    }

    //printf("%f from gpu \n",C[i].x);

}

void initgpu()
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("GPU num for max %d \n",devCount);
	cudaSetDevice(0);
}



double * gpusetdata_double(double *d_data,int N ,double *c_data)
{
	HANDLE_ERROR (cudaMalloc((void**) &d_data, N * sizeof(double)));
	HANDLE_ERROR ( cudaMemcpy(d_data, c_data, N * sizeof(double),cudaMemcpyHostToDevice));
	return d_data;
}
void gpusetdata_double_void(double *d_data,int N ,double *c_data)
{
	HANDLE_ERROR (cudaMalloc((void**) &d_data, N * sizeof(double)));
	HANDLE_ERROR ( cudaMemcpy(d_data, c_data, N * sizeof(double),cudaMemcpyHostToDevice));
}


void vector_Multi(double *data1, double *data2, cufftComplex *res,int numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorMulti<<<blocksPerGrid, threadsPerBlock>>>(data1, data2, res, numElements);
}

void cpugetdata(cufftComplex *d_outData, cufftComplex *c_outData, int N)
{
	HANDLE_ERROR ( cudaMemcpy(c_outData, d_outData , N * sizeof(cufftComplex),cudaMemcpyDeviceToHost));
}
cufftComplex* gpumallocdata(cufftComplex *d_outData,int N)
{
	HANDLE_ERROR( cudaMalloc((void**) &d_outData,  N * sizeof(cufftComplex)));
	return d_outData;
}

void printdatatofile(cufftComplex *data,int N)
{
	FILE *fp= fopen("data1.out","w+");
	for(int i=0;i< N ;i++)
	{
		fprintf(fp,"%f %f |",data[i].x,data[i].y);
		if(i%100==0)
			fprintf(fp,"\n");
	}
	fclose(fp);
}

void printdata(cufftComplex *data,int N)
{

	for(int i=0;i< N ;i++)
	{
		printf("%f %f |",data[i].x,data[i].y);
		if(i%100==0)
			printf("\n");
	}

}


int main()
{

	initgpu();
	cufftComplex *d_inputData, *d_outData;
	int nzyxdim=2000;
	double *d_Fnewweight;
	double *d_Fweight;
	double *c_data;

	double * Fnewweightdata, *Fweightdata;
	Fnewweightdata =(double*)malloc(nzyxdim*sizeof(double));
	Fweightdata =(double*)malloc(nzyxdim*sizeof(double));
	c_data= (double*)malloc(nzyxdim*sizeof(double));
	for(int i=0;i<nzyxdim;i++)
	{
		Fnewweightdata[i]=1;
		Fweightdata[i]=2;
	}
	d_Fnewweight = gpusetdata_double(d_Fnewweight,nzyxdim,Fnewweightdata);
	d_Fweight= gpusetdata_double(d_Fweight,nzyxdim,Fweightdata);


	int Fconvnum=nzyxdim;
	d_outData = gpumallocdata(d_outData,Fconvnum);
	printf("Fconvnum  : %d \n",Fconvnum);

	cufftComplex *c_output = (cufftComplex*) malloc(Fconvnum * sizeof(cufftComplex));
	vector_Multi(d_Fnewweight,d_Fweight,d_outData,nzyxdim);
	cpugetdata(d_outData,c_output,nzyxdim);

	printdata(c_output,100);
	return 0;

}