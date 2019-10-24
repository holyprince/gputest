#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"

#define NDIM 3
#define NX 3
#define NY 4
#define NZ 4

/*
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
}*/
void layoutxy(cufftComplex *data,cufftComplex *data2)
{
	int rawY= NY/2+1;

	for (int x = 0; x < NX; x++)
		for (int y = 0; y < rawY; y++) {

			data2[x*NY+y].x=data[x*rawY+y].x;
			data2[x*NY+y].y=data[x*rawY+y].y;
		}
	for (int x = 0; x < NX; x++)
		for (int y = rawY; y < NY; y++) {
			int desx,desy;
			if (x == 0)
				desx = 0;
			else
				desx = NX - x;
			desy=NY - y;

			data2[x * NY + y].x=data2[desx * NY + desy].x;
			data2[x * NY + y].y= - data2[desx * NY + desy].y;
		}


}



int main()
{


	int N[2];
	N[0] = NX; N[1] = NY;
	int LENGTH = N[0] * N[1] ;

	cufftReal *input = (cufftReal*) malloc(LENGTH * sizeof(cufftReal));
	cufftComplex *inputcccc = (cufftComplex*) malloc(LENGTH * sizeof(cufftComplex));
	cufftComplex *output_data = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	cufftComplex *output_data2 = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	cufftReal *outputreal = (cufftReal*) malloc(LENGTH * sizeof(cufftReal));

	int i;
	for (i = 0; i < LENGTH; i++) {
		input[i] = i * i +i ;
		inputcccc[i].x=i * i +i ;
		inputcccc[i].y=0;
	}



	cufftReal *d_inputData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftReal));
	cudaMemcpy(d_inputData, input, LENGTH * sizeof(cufftReal),cudaMemcpyHostToDevice);

	cufftComplex *d_inputCom;
	cudaMalloc((void**) &d_inputCom, LENGTH * sizeof(cufftComplex));
	cudaMemcpy(d_inputCom, inputcccc, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);


	cufftComplex *d_output ;
	cudaMalloc((void**) &d_output, LENGTH * sizeof(cufftComplex));
	cufftComplex *d_output2 ;
	cudaMalloc((void**) &d_output2, LENGTH * sizeof(cufftComplex));

	cufftHandle plan1,plan2,plan3,plan4;

	int t= cufftPlan2d(&plan1, N[0], N[1], CUFFT_R2C);
	t= cufftPlan2d(&plan4, N[0], N[1], CUFFT_C2C);


	cufftExecR2C(plan1, d_inputData, d_output);



	cudaMemcpy(output_data, d_output, N[0]*(N[1]/2+1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	printf("=======R2C========\n");
	for (i = 0; i < N[0]*(N[1]/2+1); i++) {
		printf("%f %f \n", output_data[i].x, output_data[i].y);
	}

	memset(output_data2,0,LENGTH*sizeof(cufftComplex));
	layoutxy(output_data,output_data2);
	printf("======after layout ====\n");
	for (i = 0; i < LENGTH; i++) {
		printf("%f %f \n", output_data2[i].x, output_data2[i].y);
	}


	cudaMemcpy(d_output2, output_data2, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	t= cufftPlan2d(&plan2, N[0], N[1], CUFFT_C2C);
	cufftExecC2C(plan2, d_output2, d_output2,CUFFT_INVERSE);
	cudaMemcpy(output_data2,d_output2 , LENGTH* sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	printf("=========================\n");
	for (i = 0; i < LENGTH; i++) {
		printf("%f %f \n", output_data2[i].x/(LENGTH), output_data2[i].y);
	}



}