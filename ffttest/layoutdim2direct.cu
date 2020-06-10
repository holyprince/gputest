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
		printf("%f ",input[i]);
		if((i+1)%NY==0)
			printf("\n");
	}



	cufftReal *d_inputData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftReal));
	cudaMemcpy(d_inputData, input, LENGTH * sizeof(cufftReal),cudaMemcpyHostToDevice);

	cufftComplex *d_inputCom;
	cudaMalloc((void**) &d_inputCom, LENGTH * sizeof(cufftComplex));
	cudaMemcpy(d_inputCom, inputcccc, LENGTH * sizeof(cufftComplex),cudaMemcpyHostToDevice);


	cufftComplex *d_output ;
	cudaMalloc((void**) &d_output, N[0]*(N[1]/2+1)  * sizeof(cufftComplex));
	cufftComplex *d_output2 ;
	cudaMalloc((void**) &d_output2, LENGTH * sizeof(cufftComplex));

	cufftHandle plan1,plan2,plan3,plan4;

	int t= cufftPlan2d(&plan1, N[0], N[1], CUFFT_R2C);
	t= cufftPlan2d(&plan4, N[0], N[1], CUFFT_C2C);


	cufftExecR2C(plan1, d_inputData, d_output);



	cudaMemcpy(output_data, d_output, N[0]*(N[1]/2+1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	printf("=======R2C========\n");
	for (i = 0; i < N[0]*(N[1]/2+1); i++) {
		printf("%f=%f ", output_data[i].x, output_data[i].y);
		if((i+1)%(NY/2+1)==0)
			printf("\n");
	}

	memset(output_data2,0,LENGTH*sizeof(cufftComplex));
	layoutxy(output_data,output_data2);
	printf("======after layout ====\n");
	for (i = 0; i < LENGTH; i++) {
		printf("%f=%f ", output_data2[i].x, output_data2[i].y);
		if((i+1)%NY==0)
			printf("\n");
	}


	cudaMemcpy(d_output2, output_data2, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	t= cufftPlan2d(&plan2, N[0], N[1], CUFFT_C2C);
	cufftExecC2C(plan2, d_output2, d_output2,CUFFT_INVERSE);
	cudaMemcpy(output_data2,d_output2 , LENGTH* sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	printf("=========================\n");
	for (i = 0; i < LENGTH; i++) {
		printf("%f=%f ", output_data2[i].x/(LENGTH), output_data2[i].y);
		if((i+1)%NY==0)
			printf("\n");
	}

/*
	0.000000 2.000000 6.000000 12.000000
	20.000000 30.000000 42.000000 56.000000
	72.000000 90.000000 110.000000 132.000000
	=======R2C========
	572.000000=0.000002 -66.000000=78.000000 -72.000000=-0.000002
	-255.999985=221.702499 10.143601=-37.856403 23.999985=-13.856392
	-256.000000=-221.702515 37.856411=-10.143597 24.000000=13.856415
	======after layout ====
	572.000000=0.000002 -66.000000=78.000000 -72.000000=-0.000002 -66.000000=-78.000000
	-255.999985=221.702499 10.143601=-37.856403 23.999985=-13.856392 37.856411=10.143597
	-256.000000=-221.702515 37.856411=-10.143597 24.000000=13.856415 10.143601=37.856403
	=========================
	0.000001=0.000008 2.000001=-0.000034 5.999997=0.000008 12.000001=-0.000034
	19.999998=-0.000004 30.000000=0.000049 42.000004=-0.000004 56.000000=0.000049
	72.000000=-0.000004 90.000000=-0.000004 110.000000=-0.000004 132.000000=-0.000004
*/
	// raw data R2C normal should C2R
/*
	cufftHandle planc2r;

	//t= cufftPlan2d(&planc2r, N[0], N[1], CUFFT_C2R);

	if (cufftPlanMany(&planc2r, 2, N,
			NULL, 1, 0,
			NULL, 1, 0,
			CUFFT_C2R, 1) != CUFFT_SUCCESS);


	cufftExecC2R(planc2r, d_output,(cufftReal *)d_output);
	cudaDeviceSynchronize();
	cudaMemcpy(outputreal, d_output, N[0]*(N[1]/2+1)  * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	printf("==========c2r===========\n");
	int linenum = (N[1]/2+1) *2;
	for (i = 0; i < N[0]*(N[1]/2+1) *2; i++) {
		printf("%f ", outputreal[i]/(LENGTH));
		if((i+1)%linenum==0)
			printf("\n");
	}
*/
	// first do C2C inverse
	cufftHandle plandiv;
	cufftPlan1d(&plandiv, N[1], CUFFT_C2C,N[0]);
	cufftExecC2C(plandiv, d_output, d_output,CUFFT_INVERSE);
	// then C2R inverse
	int dim[1]; dim[0]=N[0];
	int inembed[2];
	int outembed[2];
	inembed[0]=N[0];
	inembed[1]=N[1];
	outembed[0]=N[0];
	outembed[1]=N[1];
	int stride= N[1];
	int distance= 1;
	cufftHandle plandivcr;
	cufftPlanMany(&plandivcr, 1, dim, inembed, stride, distance, outembed, stride, distance,CUFFT_C2R, N[1]);
	cufftExecC2R(plandivcr, d_output,(cufftReal *)d_output);
	cudaDeviceSynchronize();
	cudaMemcpy(outputreal, d_output, N[0]*(N[1]/2+1)  * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	printf("==========c2r===========\n");
	int linenum = (N[1]/2+1) *2;
	for (i = 0; i < N[0]*(N[1]/2+1) *2; i++) {
		printf("%f ", outputreal[i]/(LENGTH));
		if((i+1)%linenum==0)
			printf("\n");
	}


}
