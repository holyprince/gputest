#include "stdio.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"



void changehertonature(cufftComplex *data)
{
	//for(int z=0;z<NZ;z++)
		for(int y=0;y<NY;y++)
			for(int x=0;x<(NX/2+1);x++)
			{
				int dexx,desy,desz;
				if(x==0)
					dexx=0;
				else
					dexx=NX-x;
				if(y==0)
					desy=0;
				else
					desy=NY-y;
				data[desy*NX+dexx].x=data[y*NX+x].x;
				data[desy*NX+dexx].y=data[y*NX+x].y;
				     [dexx][desy].x=data[x][y][z].x;
				data[dexx][desy].y=-data[x][y][z].y;
			}
}

int main() {

	int N[2];
	N[0]=4;N[1]=4;

	cufftReal *d_real_data,*c_real_data;
	c_real_data = (cufftReal*) malloc(sizeof(cufftReal)*N[0]*N[1]);
    cudaMalloc((void**)&d_real_data, N[0]*N[1] * sizeof(cufftReal));
	c_real_data[0]=0;
	c_real_data[1]=2;
	c_real_data[2]=4;
	c_real_data[3]=6;
	c_real_data[4]=1;
	c_real_data[5]=3;
	c_real_data[6]=5;
	c_real_data[7]=7;
	c_real_data[8]=4;
	c_real_data[9]=4;
	c_real_data[10]=4;
	c_real_data[11]=4;
	c_real_data[1]=4;

	cudaMemcpy(d_real_data,c_real_data,sizeof(cufftReal)*N[0]*N[1],cudaMemcpyHostToDevice);

	cufftComplex *d_comp_data,*c_comp_data;
	cudaMalloc((void**)&d_comp_data, sizeof(cufftComplex)*N[0]*(N[1]/2+1));
	c_comp_data = (cufftComplex*) malloc(sizeof(cufftComplex)*N[0]*(N[1]/2+1));

	cufftHandle cufftForwrdHandle;
	cufftPlan2d(&cufftForwrdHandle, N[0], N[1] , CUFFT_R2C);
	cufftExecR2C(cufftForwrdHandle, d_real_data, d_comp_data);
	cudaDeviceSynchronize();
	cudaMemcpy(c_comp_data,d_comp_data,sizeof(cufftComplex)*N[0]*(N[1]/2+1),cudaMemcpyDeviceToHost);
	cufftDestroy(cufftForwrdHandle);

	for(int i=0;i<N[0]*(N[1]/2+1);i++)
		printf("%f %f \n",c_comp_data[i].x,c_comp_data[i].y);

// C2R

	cufftHandle cufftInverseHandle;
	cudaMemset(d_real_data,0,N[0]*N[1]*sizeof(cufftReal));
	cufftPlan2d(&cufftInverseHandle, N[0],N[1], CUFFT_C2R);
	cufftExecC2R(cufftInverseHandle, d_comp_data,d_real_data);
	cudaMemcpy(c_real_data,d_real_data,sizeof(cufftReal)*N[0]*N[1],cudaMemcpyDeviceToHost);
	for(int i=0;i<N[0]*N[1];i++)
		printf("%f ",c_real_data[i]);

//C2C
	cufftComplex *d_in_data,*c_in_data;
	cudaMalloc((void**)&d_in_data, sizeof(cufftComplex)*N[0]*N[1]);
	c_in_data = (cufftComplex*) malloc(sizeof(cufftComplex)*N[0]*N[1]);

	c_in_data[0].x=0;	c_in_data[0].y=0;
	c_in_data[1].x=2;	c_in_data[1].y=0;
	c_in_data[2].x=4;	c_in_data[2].y=0;
	c_in_data[3].x=6;	c_in_data[3].y=0;
	c_in_data[4].x=1;	c_in_data[4].y=0;
	c_in_data[5].x=3;	c_in_data[5].y=0;
	c_in_data[6].x=5;	c_in_data[6].y=0;
	c_in_data[7].x=7;	c_in_data[7].y=0;
	c_in_data[8].x=4;	c_in_data[8].y=0;

	cudaMemcpy(d_in_data,c_in_data,sizeof(cufftComplex)*N[0]*N[1],cudaMemcpyHostToDevice);

	cufftHandle CCForwrdHandle;
	cufftPlan2d(&CCForwrdHandle, N[0], N[1] , CUFFT_C2C);
	cufftExecC2C(CCForwrdHandle, d_in_data, d_in_data,CUFFT_FORWARD);
	cudaDeviceSynchronize();
	cudaMemcpy(c_in_data,d_in_data,sizeof(cufftComplex)*N[0]*N[1],cudaMemcpyDeviceToHost);
	cufftDestroy(CCForwrdHandle);

	printf("\n");
	for(int i=0;i<N[0]*N[1];i++)
		printf("%f %f \n",c_in_data[i].x,c_in_data[i].y);




}
