#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"

#define NDIM 3
#define NX 3
#define NY 4
#define NZ 5
#define COMLEN N[2]*N[0]*(N[1]/2+1)


void layoutxyz(cufftComplex *data,cufftComplex *data2)
{
	int rawY= NY/2+1;


	for(int z=0;z< NZ;z++)
		for (int x = 0; x < NX; x++)
		{
			memcpy(data2+z*NX*NY+x*NY,data+z*NX*rawY+x*rawY,rawY*sizeof(cufftComplex));
		}
/*	for(int z=0;z< NZ;z++)
	for (int x = 0; x < NX; x++)
		for (int y = 0; y < rawY; y++) {

			data2[z*NX*NY+x*NY+y].x=data[z*NX*rawY+x*rawY+y].x;
			data2[z*NX*NY+x*NY+y].y=data[z*NX*rawY+x*rawY+y].y;
		}*/
	for(int z=0;z< NZ;z++)
	for (int x = 0; x < NX; x++)
		for (int y = rawY; y < NY; y++) {
			int desx,desy,desz;
			if (x == 0)
				desx = 0;
			else
				desx = NX - x;
			if(z==0)
				desz =0;
			else
				desz = NZ-z;
			desy=NY - y;

			data2[z*NX*NY+x * NY + y].x=data2[desz*NX*NY+desx * NY + desy].x;
			data2[z*NX*NY+x * NY + y].y= - data2[desz*NX*NY+desx * NY + desy].y;
		}


}



int main()
{


	int N[3];
	N[0] = NX; N[1] = NY; N[2] = NZ;
	int LENGTH = N[0] * N[1] *N[2];
	cufftReal *input = (cufftReal*) malloc(LENGTH * sizeof(cufftReal));
	cufftComplex *inputcccc = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output_data = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	cufftComplex *output_data2 = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	int i;
	for (i = 0; i < LENGTH; i++) {
		input[i] = i * i +i ;
		inputcccc[i].x= i * i +i ;
		inputcccc[i].y=0;
	}


	cufftComplex *d_inputCom;
	cudaMalloc((void**) &d_inputCom, LENGTH * sizeof(cufftComplex));
	cudaMemcpy(d_inputCom, inputcccc, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);


	cufftComplex *d_output ;
	cudaMalloc((void**) &d_output, LENGTH * sizeof(cufftComplex));


	cufftReal *d_inputData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftReal));
	cudaMemcpy(d_inputData, input, LENGTH * sizeof(cufftReal),cudaMemcpyHostToDevice);

	cufftHandle plan1,plan2;

	cufftPlan3d(&plan1, N[2], N[0], N[1], CUFFT_R2C);
	cufftPlan3d(&plan2, N[2], N[0], N[1], CUFFT_C2C);
	cufftExecR2C(plan1, d_inputData, d_output);

	cudaMemcpy(output_data, d_output, COMLEN * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	for (i = 0; i < COMLEN; i++) {
		printf("%f %f \n", output_data[i].x, output_data[i].y);
	}
	layoutxyz(output_data,output_data2);
	printf("=======output_data2========\n");
	for (i = 0; i < LENGTH; i++) {
		printf("%f %f \n", output_data2[i].x, output_data2[i].y);
	}

	cudaMemcpy(d_output, output_data2, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftExecC2C(plan2, d_output, d_output,CUFFT_INVERSE);
	cudaMemcpy(output_data, d_output, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	printf("=======C2C========\n");
	for (i = 0; i < LENGTH; i++) {
		printf("%f %f \n", output_data[i].x/LENGTH, output_data[i].y);
	}

}