#include <iostream>
using namespace std;

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>

// cuda
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cublasLt.h>

#define Value   127
#define checkCudaAPIErrors(F) if ((F) != cudaSuccess) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }

int roundoff(int v, int d)
{
    return ((v+d-1)/d) * d;
}

void matDisplayRowMajor(int rows, int cols, int8_t *p, int ld)
{
    for (int r=0; r<rows; r++)
    {
        for (int c=0; c<cols; c++)
        {
            int index = c + r * ld;

            printf("%5d", p[index]);    
        }
        printf("\n");
    }
}
void matDisplay(int rows, int cols, int8_t *p, int ld)
{
    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int index = r + c * ld;

            printf("%4d", p[index]);    
        }
        printf("\n");
    }
}

// B is in column-major
void transform2Col4(int8_t *B, int rows, int cols, int8_t *Btransform)
{
    int padding_rows = roundoff(rows, 32);
    int padding_cols = roundoff(cols, 8);

    int tile_x_32x8 = padding_cols/8;
    int tile_y_32x8 = padding_rows/32;

    for (int ty=0; ty<tile_y_32x8; ty++)
    {
        for (int tx=0; tx<tile_x_32x8; tx++)
        {
            int tile_begin = ty*32*padding_cols + tx * 32 * 8; 

            // dealing each tile of 32x8
            // the 32 rows are break into 8 groups with each 4 rows
            for (int row_group_id=0; row_group_id<8; row_group_id++)
            {
                // 4 even columns first
                for (int col_in_even=0; col_in_even<4; col_in_even++)
                {
                    // row id in each group
                    for (int innerRow_id=0; innerRow_id<4; innerRow_id++) 
                    {
                        int row_id = innerRow_id + row_group_id*4 + 32 * ty;
                        int col_id = col_in_even *2 + tx * 8; // even columns

                        int index = row_id + col_id * rows;

                        int index_col4 = innerRow_id
                                       + 4  * col_in_even
                                       + 16 * row_group_id
                                       + tile_begin;

                        if (row_id >= rows || col_id >= cols)
                        {
                            Btransform[index_col4] = 0;                            
                        }
                        else
                        {
                            Btransform[index_col4] = B[index];
                        }
                    }
                }
            }

            for (int row_group_id=0; row_group_id<8; row_group_id++)
            {
                // 4 odd columns first
                for (int col_in_odd=0; col_in_odd<4; col_in_odd++)
                {
                    // row id in each group
                    for (int innerRow_id=0; innerRow_id<4; innerRow_id++) 
                    {
                        int row_id = innerRow_id + row_group_id*4 + 32 * ty;
                        int col_id = (col_in_odd*2+1) + tx * 8; // odd columns

                        int index = row_id + col_id * rows;

                        int index_col4 = innerRow_id
                                       + 4  * col_in_odd
                                       + 16 * (row_group_id+8) // be careful here
                                       + tile_begin;

                        if (row_id >= rows || col_id >= cols)
                        {
                            Btransform[index_col4] = 0;                            
                        }
                        else
                        {
                            Btransform[index_col4] = B[index];
                        }
                    }
                }
            }
        }
    }
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define checkcuBlasError(F) if ((F) != CUBLAS_STATUS_SUCCESS) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, _cudaGetErrorEnum(F)); exit(-1); }

int ltIgemmTensor(cublasLtHandle_t ltHandle,
        int m, int n, int k,
        const int8_t *A,
        int lda,
        const int8_t *B,
        int ldb,
        int32_t *C,
        int ldc,
        int iters,
        float &time_matmul)
{
    cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc = NULL;
    cublasLtMatrixLayout_t Bdesc = NULL;
    cublasLtMatrixLayout_t Cdesc = NULL;

    int32_t alpha = 1;
    int32_t beta  = 0;

    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // The tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t  *Atransform = NULL;
    int8_t  *Btransform = NULL;
    int32_t *Ctransform = NULL;

    cublasLtMatrixLayout_t AtransformDesc = NULL;
    cublasLtMatrixLayout_t BtransformDesc = NULL;
    cublasLtMatrixLayout_t CtransformDesc = NULL;

    float transformAlpha = 1.0f;
    float transformBeta  = 0.0f;

    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldaTransform = 32 * m;
    int ldbTransform = 32 * roundoff(n,8);
    int ldcTransform = 32 * m;

    checkCudaAPIErrors(cudaMalloc((void **)&Atransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldaTransform));
    checkCudaAPIErrors(cudaMalloc((void **)&Btransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldbTransform));
    checkCudaAPIErrors(cudaMalloc((void **)&Ctransform, sizeof(int32_t )* roundoff(n, 32)/32*ldcTransform));

    // create transformDesc
    checkcuBlasError(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    // create matmulDesc
    checkcuBlasError(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32I));

    // Tensor op igemm kernels only support NT gemm
    checkcuBlasError(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t)));

    // Create descriptors for the original matrices
    checkcuBlasError(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I,  m, k, lda));
    // note the B matrix is transposed. 
    checkcuBlasError(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I,  n, k, ldb)); 
    checkcuBlasError(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc)); 

    // Create descriptors for the transformed matrices
    checkcuBlasError(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I,  m, k, ldaTransform));
    checkcuBlasError(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32)));
     
    checkcuBlasError(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I,  n, k, ldbTransform));
    checkcuBlasError(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    checkcuBlasError(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform));
    checkcuBlasError(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32)));

#ifdef DEBUG_
    
   int8_t *h_A  = NULL; 
   int8_t *h_Atransform = NULL; 
   int8_t *h_Btransform = NULL;

   h_A = (int8_t* )malloc(sizeof(int8_t ) * m * k);
   h_Atransform = (int8_t* )malloc(sizeof(int8_t ) * roundoff(k, 32)/32*ldaTransform);
   h_Btransform = (int8_t* )malloc(sizeof(int8_t ) * roundoff(k, 32)/32*ldbTransform);


   checkCudaAPIErrors(cudaMemcpy(h_A, A, sizeof(int8_t) * m * k, cudaMemcpyDeviceToHost));

#endif 

int transformColumnMajor2COl32(int rows, int cols,
                               const int8_t *A, int lda,
                               int8_t *ATransform, int ldaTransform);
int transformCOl322ColumnMajor(int rows, int cols,
                               int *CTransform, int ldcTransform,
                               int *C,          int ldc);

#ifdef CUSTOMERIZED_KERNEL 
    // transform with customized kernel
    for (int i=0; i<iters; i++)
    {
    transformColumnMajor2COl32(m, k, 
                               A, lda,
                               Atransform, ldaTransform);
    }
#ifdef DEBUG_
   checkCudaAPIErrors(cudaMemcpy(h_Atransform, Atransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldaTransform, cudaMemcpyDeviceToHost));
#endif 
#else
    for (int i=0; i<iters; i++)
    checkcuBlasError(cublasLtMatrixTransform(ltHandle, 
                transformDesc, 
                &transformAlpha,
                A, Adesc,
                &transformBeta,
                NULL, NULL, 
                Atransform, AtransformDesc, 0));
#ifdef DEBUG_
   checkCudaAPIErrors(cudaMemcpy(h_Atransform, Atransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldaTransform, cudaMemcpyDeviceToHost));
#endif
#endif //CUSTOMERIZED_KERNEL 

#ifdef DEBUG_
   char filenameA[] = "A.txt"; 
   char filenameAtransform[] = "Atransform.txt"; 

    FILE *outputA = NULL;
    FILE *outputAtransform = NULL;

	if ((outputA = fopen(filenameA, "w")) == NULL)
	{
		printf("Can not open file : %s\n", filenameA);
		exit(1);
	}

	if ((outputAtransform = fopen(filenameAtransform, "w")) == NULL)
	{
		printf("Can not open file : %s\n", filenameAtransform);
		exit(1);
	}

    // write A to file
    int x, y;
    int index;
    for (x=0; x<k; x++)
    {
        for (y=0; y<lda; y++)
        {
            index = x*lda + y;
            fprintf(outputA, "%3d", h_A[index]);
        }
        fprintf(outputA, "\n");
    }

    for (x=0; x<roundoff(k, 32)/32; x++)
    {
        for (y=0; y<ldaTransform; y++)
        {
            index = x*ldaTransform + y;
            fprintf(outputAtransform, "%3d", h_Atransform[index]);
        }
        fprintf(outputAtransform, "\n");
    }

    fclose(outputA);
    fclose(outputAtransform);

    free(h_A);
    free(h_Atransform);
#endif 

    checkcuBlasError(cublasLtMatrixTransform(ltHandle, 
                transformDesc, 
                &transformAlpha,
                B, Bdesc,
                &transformBeta,
                NULL, NULL, 
                Btransform, BtransformDesc, 0));

    cudaEventRecord(start, 0);
    for (int i=0; i<iters; i++)
    {
        checkcuBlasError(cublasLtMatmul(ltHandle, 
                    matmulDesc,
                    &alpha,
                    Atransform,
                    AtransformDesc,
                    Btransform,
                    BtransformDesc,
                    &beta,
                    Ctransform,
                    CtransformDesc,
                    Ctransform,
                    CtransformDesc,
                    NULL, NULL, 0, 0));

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_matmul, start, stop);

#ifdef CUSTOMERIZED_KERNEL 
    // transform with customized kernel
    for (int i=0; i<iters; i++)
    {
    transformCOl322ColumnMajor(m, n, 
                               Ctransform, ldcTransform,
                               C,          ldc);
    }
#else
    for (int i=0; i<iters; i++)
    checkcuBlasError(cublasLtMatrixTransform(ltHandle, 
                transformDesc,
                &transformAlpha,
                Ctransform,
                CtransformDesc, 
                &transformBeta,
                NULL, NULL, 
                C, Cdesc, 0));
#endif //CUSTOMERIZED_KERNEL

#ifdef DEBUG_
   checkCudaAPIErrors(cudaMemcpy(h_Btransform, Btransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldbTransform, cudaMemcpyDeviceToHost));

   printf("display Btransform with gpu transform\n");
   matDisplay(ldbTransform, roundoff(k, 32)/32, h_Btransform, ldbTransform);

   free(h_Btransform);
#endif

    time_matmul /= iters;

    // Descriptors are no longer needed as all GPU work was already enqueued.
    if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
    if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
    if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
    if (transformDesc) cublasLtMatrixTransformDescDestroy(transformDesc);

    // Wait until device is done before freeing transformed buffers
    cudaDeviceSynchronize();
    if (Ctransform) cudaFree(Ctransform);
    if (Btransform) cudaFree(Btransform);
    if (Atransform) cudaFree(Atransform);

    return cublasStat == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

// initialize matrix in column-major
void matInit(int rows, int cols, int8_t *p, int ld)
{
    srand(time(NULL)); 

    int i = 0;

    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int index = r + c * ld;
            
            p[index] = rand()%10;
            
            //p[index] = (i++);
        }
    }
}



// mat  is column-major
// matT is row-major 
void transpose(int8_t *matT, int8_t *mat, int rows, int cols)
{
    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int indexIn = r + c*rows;
            int indexOut= c + r*cols;

            matT[indexOut] = mat[indexIn];
        }
    }
}

void matMul(int m, int n, int k,
        const int8_t *A,
        int lda,
        const int8_t *B,
        int ldb,
        int32_t *C,
        int ldc)
{
    int32_t sum;

    for (int c=0; c<n; c++)
    {
        for (int r=0; r<m; r++)
        {
            sum = 0; 

            for (int kk=0; kk<k; kk++)
            {
                int idxA = kk*lda + r; // A[r][kk]       
                int idxB = c*ldb + kk; // B[kk][c]

                sum += A[idxA] * B[idxB];
            }

            C[c*ldc + r] = sum; // C[r][c]
        }
    }
}

// Input matrices are column-major
// Only output the first different element per column
void postprocess(const int32_t *ref, const int32_t *res, int m, int n, int k, float ms)
{
    bool passed = true;

    for (int c=0; c<n; c++)
    {
        for (int r=0; r<m; r++)
        {
            int index = r + c*m;

            if (ref[index] !=  res[index])
            {
                printf("(row = %d, col = %d) gpu result=%d cpu ref=%d  ", r, c, res[index], ref[index]);
                printf("%25s\n", "*** FAILED ***");
                passed = false;
                break;
            }
        }
    }
  if (passed)
  {
    float TFlops = (long(2))*m*n*k/(ms * 1000 * 1000 * 1000); // unit in Tflops

    printf("%6d\t%6d\t%6d\t%10.6f\t%10.6f\n", m, k, n, ms, TFlops);
  }
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        fprintf(stderr, "gemm_imma m n k iters devID\n");
        return -1;
    }

    int32_t alpha = 1;
    int32_t beta  = 0;

    float TFlops;

    int m = 1024; 
    int k = 1024;
    int n = 1024;
    int iters = 100;
    int devID = 0;

    int8_t  *h_A = NULL; 	// m * k, stored in column-major
    int8_t  *h_B = NULL; 	// k * n, stored in column-major
    int8_t  *h_BT = NULL; 	// k * n, stored in column-major
    int32_t *h_C = NULL; 	// m * n, stored in column-major
    int32_t *h_Cres = NULL; // m * n, stored in column-major

    int8_t  *d_A = NULL; // m * k, stored in column-major
    int8_t  *d_B = NULL; // k * n, stored in column-major
    int8_t  *d_BT= NULL; // k * n, stored in column-major, also think B saved in row-major
    int32_t *d_C = NULL; // m * n, stored in column-major

    // get params from input args
    m     = atoi(argv[1]);
    n     = atoi(argv[2]);
    k     = atoi(argv[3]);
    iters = atoi(argv[4]);
    devID = atoi(argv[5]);

    cudaSetDevice(devID);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devID);
    printf("Device : %s, compute SM %d.%d.\n",devProp.name, devProp.major, devProp.minor);

    // allocate memory
    h_A = (int8_t* )malloc(sizeof(int8_t ) * m * k);
    if (!h_A) printf("falied to allocate mem on CPU");
    h_B = (int8_t* )malloc(sizeof(int8_t ) * k * n);   // B : k*n
    if (!h_B) printf("falied to allocate mem on CPU"); // BT: n*k, the transpose of B
    h_BT= (int8_t* )malloc(sizeof(int8_t ) * n * k);
    if (!h_BT) printf("falied to allocate mem on CPU");
    h_C = (int32_t*)malloc(sizeof(int32_t) * m * n);
    if (!h_C) printf("falied to allocate mem on CPU");
    h_Cres = (int32_t*)malloc(sizeof(int32_t) * m * n);
    if (!h_Cres) printf("falied to allocate mem on CPU");

    int8_t *h_Btransform = NULL;
    int ldbTransform = 32 * roundoff(n,8);
    h_Btransform = (int8_t* )malloc(sizeof(int8_t ) * roundoff(k, 32)/32*ldbTransform);
    if (!h_Btransform) printf("falied to allocate mem on CPU");

    checkCudaAPIErrors(cudaMalloc((void **)&d_A, sizeof(int8_t ) * m * k));
    checkCudaAPIErrors(cudaMalloc((void **)&d_B, sizeof(int8_t ) * k * n));
    checkCudaAPIErrors(cudaMalloc((void **)&d_BT,sizeof(int8_t ) * n * k));
    checkCudaAPIErrors(cudaMalloc((void **)&d_C, sizeof(int32_t) * m * n));

    cublasLtHandle_t ltHandle;
    checkcuBlasError(cublasLtCreate(&ltHandle));

	cublasHandle_t handle;
	checkcuBlasError(cublasCreate(&handle));

	float time_used  = 0.0; // ms
	float time_matmul= 0.0; // ms

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cublasStatus_t cublasStat;

    printf("initialize A and B with m=%d, n=%d and k=%d\n", m, n, k);
    matInit(m, k, h_A, m);
    matInit(k, n, h_B, k);
    transpose(h_BT, h_B, k, n);
    
	// transform the B matrix from column-major to col4 format
	// for the details of col4 format, please refers to the doc of cublasLt
    transform2Col4(h_B, k, n, h_Btransform);

#ifdef DEBUG_
    printf("display B\n");
    matDisplay(k, n, h_B, k);
    printf("display BT\n");
    matDisplay(n, k, h_BT, n);
    printf("display Btransform with cpu transform\n");
    matDisplay(ldbTransform, roundoff(k, 32)/32, h_Btransform, ldbTransform);
#endif //DEBUG_

    printf("do the gemm on CPU as a reference result\n");
    matMul(m, n, k, h_A, m, h_B, k, h_C, m);

    printf("copy date from host to device\n");
    matMul(m, n, k, h_A, m, h_B, k, h_C, m);
    checkCudaAPIErrors(cudaMemcpy(d_A, h_A, sizeof(int8_t) * m * k,cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_B, h_B, sizeof(int8_t) * k * n,cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_BT,h_BT,sizeof(int8_t) * n * k,cudaMemcpyHostToDevice));

    printf("do the gemm on GPU with cublasGemmEx\n");
    cudaEventRecord(start, 0);

    for (int t = 0; t < iters; t++)
    {

        cublasStat=cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, 
                                &alpha, 
                                d_A, CUDA_R_8I, m, 
                                d_B, CUDA_R_8I, k, 
                                &beta, 
                                d_C, CUDA_R_32I, m,
                                CUDA_R_32I,				
                static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if(cublasStat != CUBLAS_STATUS_SUCCESS)
        {
            checkcuBlasError(cublasStat);
            continue;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_used, start, stop);

    time_used /= iters;

    checkCudaAPIErrors(cudaMemcpy(h_Cres, d_C, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost));
    postprocess(h_C, h_Cres, m, n, k, time_used);

    // cublasLtMatMul for gemm
    printf("do the gemm on GPU with ltIgemmTensor\n");
    cudaEventRecord(start, 0);

    ltIgemmTensor(ltHandle,
                  m, n, k,
                  d_A, m,
                  d_BT,n,
                  d_C, m,
                  iters,
                  time_matmul);

    checkCudaAPIErrors(cudaMemcpy(h_Cres, d_C, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost));
    postprocess(h_C, h_Cres, m, n, k, time_matmul);

    free(h_A);
    free(h_B);
    free(h_BT);
    free(h_C);
    free(h_Cres);
    free(h_Btransform);
    checkCudaAPIErrors(cudaFree(d_A));
    checkCudaAPIErrors(cudaFree(d_B));
    checkCudaAPIErrors(cudaFree(d_BT));
    checkCudaAPIErrors(cudaFree(d_C));

	checkCudaAPIErrors(cudaEventDestroy(start));
	checkCudaAPIErrors(cudaEventDestroy(stop));
	checkcuBlasError(cublasDestroy(handle));
    checkcuBlasError(cublasLtDestroy(ltHandle));
}
