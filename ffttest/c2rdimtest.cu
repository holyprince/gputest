#include <stdio.h>
#include "cufft.h"

#define NX 3
#define NY 3
#define LENGTH NX*NY
#define LENGTHC (NX/2+1)*NY
int main()
{



	float data[9];  //,data2[9];
	int i;
	//data[0]=1;data[1]=2;data[2]=3;data[3]=4;data[4]=-3;data[5]=6;

	for(int i=0;i<NX*NY;i++)
	{
		data[i]=i+2;
		printf("%f ",data[i]);
		if(i%3==2)
			printf("\n");
	}

	cufftComplex *output_data = (cufftComplex*) malloc( LENGTH * sizeof(cufftComplex));
	cufftReal *output_datareal = (cufftReal*) malloc( LENGTH * sizeof(cufftReal));

	cufftReal *d_inputData;
	cudaMalloc((void**) &d_inputData, LENGTH * sizeof(cufftReal));
	cudaMemcpy(d_inputData, data, LENGTH * sizeof(cufftReal),cudaMemcpyHostToDevice);

	cufftComplex *d_output ;
	cudaMalloc((void**) &d_output, LENGTH * sizeof(cufftComplex));

/*	cufftHandle plan1,planc2r;

	int t= cufftPlan2d(&plan1, NX, NY, CUFFT_R2C);
	cufftExecR2C(plan1, d_inputData, d_output);

	cudaMemcpy(output_data, d_output, (NX/2+1)*NY * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (i = 0; i < (NX/2+1)*NY; i++) {
		printf("%f %f\n", output_data[i].x, output_data[i].y);
	}

	t= cufftPlan2d(&planc2r, NX, NY, CUFFT_C2R);
	cufftExecC2R(planc2r, d_output, (cufftReal *)d_output);

	cudaMemcpy(output_datareal, d_output, NX*NY * sizeof(cufftReal), cudaMemcpyDeviceToHost);

	for (i = 0; i < NX*NY; i++) {
		printf("%f \n", output_datareal[i]/(NX*NY));
	}
*/
	cufftHandle plan1,plan2;
	size_t workSize[2];

/*
	int rank=1;
	int n[1];
	n[0]=NX;
	int istride=1;
	int idist = NX;
	int ostride=1;
	int odist = NX;
	int inembed[2];
	int onembed[2];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[0] = NY;
	cufftPlanMany(&plan1,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, NY);
	cufftExecR2C(plan1, d_inputData, d_output);
	rank=1;
	n[0]=NY;
	istride=NX;
	idist = 1;
	ostride=NX;
	odist = 1;
	inembed[2];
	onembed[2];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[0] = NY;

	cufftPlanMany(&plan2,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2C, NX);

	cufftExecC2C(plan2, d_output, d_output,CUFFT_FORWARD);

	cudaMemcpy(output_data, d_output, NX*NY * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	for (i = 0; i < NX*NY; i++) {
		printf("%f %f\n", output_data[i].x, output_data[i].y);
	}
*/

	int rank=1;
	int n[1];
	n[0]=NY;
	int istride=NX;
	int idist = 1;
	int ostride=NX;
	int odist = 1;
	int inembed[2];
	int onembed[2];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[0] = NY;
	cufftPlanMany(&plan2,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, NX);

	cufftExecR2C(plan2, d_inputData, d_output);



	rank=1;

	n[0]=NX;
	istride=1;
	idist = NX;
	ostride=1;
	odist = NX;
	inembed[2];
	onembed[2];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[0] = NY;
	cufftPlanMany(&plan1,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2C, NY);
	cufftExecC2C(plan1, d_output, d_output,CUFFT_FORWARD);




	cudaMemcpy(output_data, d_output, NX*NY * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	for (i = 0; i < NX*NY; i++) {
		printf("%f %f\n", output_data[i].x, output_data[i].y);
	}



	printf("data: \n%f  \n",(data+3)[0]);

	return 0;
}

/*
 *
9.000000 -0.000000
-1.500000 0.866025
18.000000 0.000000
-1.499999 0.866025
27.000000 -0.000000
-1.499999 0.866025
 */
