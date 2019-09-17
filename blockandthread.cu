int nx = 1 << 14;  
int ny = 1 << 14;  
int dimx = 32;  
dim3 block(dimx, 1);  
dim3 grid((nx + block.x - 1) / block.x, 1); 


__global__ void kernel_function()  
{  
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;  
  
    if (ix < nx )  
    {  
        do_something();  
    }  
}  


int nx = 1 << 14;  
int ny = 1 << 14;  
int dimx = 32;  
dim3 block(dimx, 1);  
dim3 grid((nx + block.x - 1) / block.x, ny); 


__global__ void kernel_function()  
{  
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    unsigned int iy = blockIdx.y;  
    unsigned int idx = iy * nx + ix;  
  
    if (ix < nx && iy < ny)  
    {  
        do_something();  
    }  
}  


int nx = 1 << 14;  
int ny = 1 << 14;  
int dimx = 32;  
int dimy = 32;  
dim3 block(dimx, dimy);  
dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);  

__global__ void kernel_function()  
{  
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;  
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;  
    unsigned int idx = iy * nx + ix;  
  
    if (ix < nx && iy < ny)  
    {  
        do_something();  
    }  
}