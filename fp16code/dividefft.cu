#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"
#include "time.h"



void testmoduleGPU(int dimx,int dimy,int dimz) {
	int N[3];
	N[0] = dimx; N[1] = dimy; N[2] = dimz;
	int LENGTH = N[0] * N[1] * N[2];
	cufftComplex *input = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*) malloc(
			LENGTH * sizeof(cufftComplex));
	int i;
	for (i = 0; i < LENGTH; i++) {
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

	cufftExecC2C(plan, d_inputData, d_outData, CUFFT_FORWARD);

	cudaMemcpy(output, d_outData, LENGTH * sizeof(cufftComplex),
			cudaMemcpyDeviceToHost);

	for (i = 0; i < 10; i++) {
		printf("%f %f \n", output[i].x, output[i].y);
	}
	for(int i=0+dimx*dimy;i<10+dimx*dimy;i++)
		printf("%f %f \n",output[i].x,output[i].y);
	cufftDestroy(plan);
	free(input);
	free(output);
	cudaFree(d_inputData);
	cudaFree(d_outData);
}


void dividefft(int dimx,int dimy,int dimz)
{
	int N[3];
	N[0] = dimx; N[1] = dimy; N[2] = dimz;
	int LENGTH = N[0] * N[1] * N[2];
	cufftComplex *cinput = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));

	for(int i=0;i<LENGTH;i++)
	{
		cinput[i].x=i% 1000;
		cinput[i].y=0;
	}

	cufftComplex *d_inputData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftComplex));

	cudaMemcpy(d_inputData, cinput, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);

	cufftHandle plan;

	// for one dimension

	int BATCH= dimy * dimz;
	cufftPlan1d(&plan, dimx , CUFFT_C2C, BATCH);
	cufftExecC2C(plan, d_inputData, d_inputData,CUFFT_FORWARD);
	cufftDestroy(plan);


	// for two dimension
	int dimn[1];
	dimn[0]=dimy;
	int inembed[2];
	inembed[0]=dimx;inembed[1]=dimy;
	int istride = dimx;
	int idist=1;
	int onembed[2];
	onembed[0]=dimx;onembed[1]=dimy;
	int ostride = dimx;
	int odist=1;
	cufftPlanMany(&plan, 1, dimn, inembed,istride, idist, onembed, ostride,odist,CUFFT_C2C, dimx);

	for(int i=0;i<dimz;i++)
		cufftExecC2C(plan, d_inputData+ i*(dimx*dimy), d_inputData + i*(dimx*dimy), CUFFT_FORWARD);
	cufftDestroy(plan);


	// for three dimension
	int dimnz[0];
	dimnz[0]=dimz;
	int inembedz[3],outembedz[3];
	inembedz[0]=dimx;inembedz[1]=dimy;inembedz[2]=dimz;
	int istridez = dimx*dimy;
	int idistz=1;
	outembedz[0]=dimx;outembedz[1]=dimy;outembedz[2]=dimz;
	int ostridez = dimx*dimy;
	int odistz=1;


	cufftPlanMany(&plan, 1, dimnz, inembedz,istridez, idistz, outembedz, ostridez,odistz, CUFFT_C2C, dimx*dimy);
	cufftExecC2C(plan, d_inputData, d_inputData, CUFFT_FORWARD);




	cudaMemcpy(cinput, d_inputData , LENGTH * sizeof(cufftComplex),cudaMemcpyDeviceToHost);


	for (int i = 0; i < 10; i++) {
		printf("%f %f \n", cinput[i].x, cinput[i].y);
	}
	for(int i=0+dimx*dimy;i<10+dimx*dimy;i++)
		printf("%f %f \n",cinput[i].x,cinput[i].y);

	cufftDestroy(plan);
	cudaFree(cinput);


}

void dividefft2d1d(int NX,int NY,int NZ)
{
    int res;
    int LENGTH= NX*NY*NZ;
    int dimx=NX; int dimy=NY; int dimz=NZ;
    cufftComplex *d_in ;

	cufftComplex *cinput = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));

	for(int i=0;i<LENGTH;i++)
	{
		cinput[i].x=i% 1000;
		cinput[i].y=0;
	}
    cudaMalloc((void**) & (d_in), sizeof(cufftComplex) * NX*NY*NZ);
    cudaMemcpy(d_in, cinput, NX*NY*NZ * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    size_t worksize=0;
    int xyN[2];
    xyN[0]=NX;
    xyN[1]=NY;
    cufftHandle xyplan;
    res=cufftPlanMany(&xyplan, 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, NZ);
    //printf("check1: %d\n",res);
    res=cufftExecC2C(xyplan, d_in, d_in, CUFFT_FORWARD);
    //printf("check2: %d\n",res);
    res=cufftGetSizeMany(xyplan,2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, NZ, &worksize);
    //printf(" worksize 1 %ld \n ",worksize);
    cufftDestroy(xyplan);
    int zN[1];
    zN[0]=NZ;
    int inembed[3];
    inembed[0]=NX;inembed[1]=NY;inembed[2]=NZ;

    cufftHandle zplan;
    res=cufftPlanMany(&zplan, 1, zN, inembed, NX*NY, 1, inembed, NX*NY, 1, CUFFT_C2C, NX*NY);
    //printf("check3: %d\n",res);
    res=cufftGetSizeMany(zplan,1,zN, inembed, NX*NY, 1, inembed, NX*NY, 1, CUFFT_C2C,NX*NY,&worksize);
    //printf(" worksize 2 %ld \n ",worksize);

    res=cufftExecC2C(zplan, d_in, d_in, CUFFT_FORWARD);
    //printf("check4: %d\n",res);
    cudaMemcpy(cinput, d_in, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++) {
		printf("%f %f \n", cinput[i].x, cinput[i].y);
	}
	for(int i=0+dimx*dimy;i<10+dimx*dimy;i++)
		printf("%f %f \n",cinput[i].x,cinput[i].y);

    cudaFree(d_in);
    cufftDestroy(zplan);
}


int main() {
	//readdataGPU(10,10,10);
	testmoduleGPU(100,100,100);
	printf("\n");
	dividefft(100,100,100);
	printf("\n");
	dividefft2d1d(100,100,100);
}


