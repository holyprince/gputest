#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"

#define NDIM 3
#define NX 3
#define NY 3
#define NZ 4


void layoutxyz(cufftComplex *data,cufftComplex *data2)
{
	int rawX=(NX/2+1);
	for(int z=0;z<NZ;z++)
		for(int y=0;y<NY;y++)
			for(int x=0;x<(NX/2+1);x++)
			{
				int desx,desy,desz;
				data2[z*NX*NY+y*NX+x].x=data[z*rawX*NY+y*rawX+x].x;
				data2[z*NX*NY+y*NX+x].y=data[z*rawX*NY+y*rawX+x].y;
				if(x==0)
					desx=0;
				else
					desx=NX-x;
				if (desx >= rawX) {
					if (y == 0)
						desy = 0;
					else
						desy = NY - y;
					data2[z*NX*NY+desy*NX+desx].x = data2[z*NX*NY+y*NX+x].x;
					data2[z*NX*NY+desy*NX+desx].y = -data2[z*NX*NY+y*NX+x].y;
				}
			}
}

void layoutxy(int *data,int *data2)
{
	int rawX=(NX/2+1);

		for(int y=0;y<NY;y++)
			for(int x=0;x<(NX/2+1);x++)
			{
				int dexx,desy,desz;
				if(x==0)
					dexx=0;
				else
					dexx=NX-x;
				if (y == 0)
					desy = 0;
				else
					desy = NY - y;
				if (dexx >= rawX) {
					data2[desy * NX + dexx] = -data[y * rawX + x];
					data2[y * NX + x] = data[y * rawX + x];
				}
				else
					data2[desy * NX + dexx] = data[y * rawX + x];
			}
}

void layout(int *data)
{
	int rawX=(NX/2+1);
// from back to start
	for(int i=(rawX*NY-1);i>=0;i--)
	{
		int y=i/NY;
		int x=i%rawX;
		data[y * rawX + x] = data[i];
	}
/*		for(int y=0;y<NY;y++)
			for(int x=0;x<(NX/2+1);x++)
			{
				data[y * NX + x] = data[y * rawX + x];
			}*/
	/*	for(int y=0;y<NY;y++)
			for(int x=(NX/2+1);x<NX;x++)
			{

				data[NY - y * NX + NX-x] = data[y * NX + x];
			}*/
}
int main()
{


	int N[3];
	N[0] = NX; N[1] = NY; N[2] = NZ;
	int LENGTH = N[0] * N[1] * N[2];

	cufftReal *input = (cufftReal*) malloc(LENGTH * sizeof(cufftReal));
	cufftComplex *output_data = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	cufftComplex *output_data2 = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	cufftReal *outputreal = (cufftReal*) malloc(LENGTH * sizeof(cufftReal));
	int i;
	for (i = 0; i < LENGTH; i++) {
		input[i] = i * i +i ;
	}

	cufftReal *d_inputData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftReal));
	cudaMemcpy(d_inputData, input, LENGTH * sizeof(cufftReal),cudaMemcpyHostToDevice);

	cufftComplex *d_output ;
	cudaMalloc((void**) &d_output, LENGTH * sizeof(cufftComplex));
	cufftComplex *d_output2 ;
	cudaMalloc((void**) &d_output2, LENGTH * sizeof(cufftComplex));

	cufftHandle plan1,plan2,plan3;

	int t= cufftPlan3d(&plan1, N[0], N[1], N[2], CUFFT_R2C);


	//cufftGetSize3d(plan,N[0], N[1], N[2],CUFFT_C2C, worksize);
	//printf("make : %d %ld \n",t,worksize[0]);
	cufftExecR2C(plan1, d_inputData, d_output);

	cudaMemcpy(output_data, d_output, (N[0]/2+1)*N[1]*N[2] * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (i = 0; i < (N[0]/2+1)*N[1]*N[2]; i++) {
		printf("%f %f \n", output_data[i].x, output_data[i].y);
	}
	layoutxyz(output_data,output_data2);
	printf("=========================\n");
	for (i = 0; i < (N[0])*N[1]*N[2]; i++) {
		printf("%f %f \n", output_data2[i].x, output_data2[i].y);
	}

	t= cufftPlan3d(&plan3, N[0], N[1], N[2], CUFFT_C2R);


	//cufftGetSize3d(plan,N[0], N[1], N[2],CUFFT_C2C, worksize);
	//printf("make : %d %ld \n",t,worksize[0]);
	cufftExecC2R(plan3, d_output, d_inputData);
	cudaMemcpy(input, d_inputData, LENGTH * sizeof(cufftReal),cudaMemcpyDeviceToHost);
	printf("=====normal C2R =========\n");
	for (i = 0; i < (N[0])*N[1]*N[2]; i++) {
		printf("%f  \n",input[i]/(N[0]*N[1]*N[2]));
	}
	//C2C

	cudaMemcpy(d_output2, output_data2, (N[0])*N[1]*N[2] * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	t= cufftPlan3d(&plan2, N[0], N[1], N[2], CUFFT_C2C);
	cufftExecC2C(plan2, d_output2, d_output2,CUFFT_INVERSE);
	cudaMemcpy(output_data2,d_output2 , (N[0])*N[1]*N[2] * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	printf("=========================\n");
	for (i = 0; i < (N[0])*N[1]*N[2]; i++) {
		printf("%f %f \n", output_data2[i].x/(N[0]*N[1]*N[2]), output_data2[i].y);
	}

/*
	layout(output_data);

	printf("After :\n");
	cudaMemcpy(d_output, output_data, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	cufftPlan3d(&plan2, N[0], N[1], N[2], CUFFT_C2C);
	cufftExecC2C(plan2, d_output, d_output,CUFFT_INVERSE);
	cudaMemcpy(output_data2, d_output, LENGTH * sizeof(cufftReal), cudaMemcpyDeviceToHost);
	for (i = 0; i < 10; i++) {
		printf("%f \n", output_data2[i].x/LENGTH);
	}





	cufftDestroy(plan1);
	cufftDestroy(plan2);
	free(input);
	free(output_data);
	cudaFree(d_inputData);
	cudaFree(d_output);*/
}
