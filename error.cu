__host__ ​ __device__ ​const char* cudaGetErrorName ( cudaError_t error )
__host__ ​ __device__ ​const char* cudaGetErrorString ( cudaError_t error )

    cudaError_t err;
    err = cudaMemcpy(p_d, p_h, sizeof(float)*1024, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }
	
#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);
//因此，这里，我们就使用这个宏来分析runtime api是否调用正确了：
checkCudaErrors( cudaMemcpy(p_d, p_h, sizeof(float)*1024, cudaMemcpyHostToDevice) );

//method2 start (more used often)
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

//method2 end

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}	



#ifdef DEBUG_HIP
#define DEBUG_HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define DEBUG_PRIVATE_ERROR(func, status) {  \
                       (status) = (func); \
                       DEBUG_HANDLE_ERROR(status); \
                   }
#else
#define DEBUG_HANDLE_ERROR( err ) (err) //Do nothing
#define DEBUG_PRIVATE_ERROR( err ) (err) //Do nothing
#endif

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define PRIVATE_ERROR(func, status) {  \
                       (status) = (func); \
                       HANDLE_ERROR(status); \
                   }

static void HandleError( hipError_t err, const char *file, int line )
{

    if (err != hipSuccess)
    {
    	fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
						hipGetErrorString( err ), file, line, err );
		fflush(stdout);
#ifdef DEBUG_HIP
		raise(SIGSEGV);
#else
		CRITICAL(ERRGPUKERN);
#endif
    }
}

//from nsight tmeplate : 

#include <cuda.h>
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

int main()
{
	CHECK_CUDA_RESULT(cuInit(0));
}

