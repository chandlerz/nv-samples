#include <stdio.h>
#include <assert.h>

const int BLOCK_DIMX = 32;
const int COL32      = 32; // BLOCM_DIMY
const int TILE_X     = 128;
const int TILE_Y     = COL32;

template<class T>
__global__ void COL322ColumnMajor(int rows, int cols,
                                  T *CTransform, int ldcTransform,
                                  T *C,          int ldc)
{
    // CTransform is in Col32 format. 
    // C is stored in column-major format
    int tx = threadIdx.x; // value in [0, 31]. column ID of elements in each 32m
    int ty = threadIdx.y; // 

    int idInBlock   = tx + ty * blockDim.x;

    // left shift 10 bits because each block deals with 32x32 elements
    // 1024 = blockDim.x * blockDim.y
    int rowID = idInBlock + (blockIdx.x<<10); 
    int colID = blockIdx.y;
    int inIndex = rowID + colID * ldcTransform;

    __shared__ T tile[COL32][COL32 + 1];

    rowID = threadIdx.y + (blockIdx.x<<5); // * 32. this 32 is because n is divided by 32

    if ((rowID < rows))
        tile[threadIdx.y][threadIdx.x] = CTransform[inIndex];

    __syncthreads();

    rowID = threadIdx.x + (blockIdx.x<<5); // * 32. this 32 is because m is divided by 32
    colID = threadIdx.y + (blockIdx.y<<5); // * 32. this 32 is because n is divided by 32

    if ((rowID < rows) && (colID < cols))
    {
        int outIndex = rowID + colID * ldc;
        C[outIndex]  = tile[threadIdx.x][threadIdx.y];
    }
}

template<class T>
__global__ void ColumnMajor2COl32(int rows, int cols,
                                           const T *A, int lda,
                                           T *ATransform, int ldaTransform)
{
    int rowID = threadIdx.x + blockDim.x * blockIdx.x;
    int colID = threadIdx.y + blockDim.y * blockIdx.y;
    int inIndex, outIndex;

    __shared__ T tile[COL32][BLOCK_DIMX + (4/sizeof(T))];

    inIndex = rowID + colID * lda;

    if (rowID < rows)
    {
        if (colID < cols)
            tile[threadIdx.y][threadIdx.x] = A[inIndex];
        else
            tile[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    int newRowID = threadIdx.x + threadIdx.y * COL32 + blockIdx.x * blockDim.x * COL32; 
    int newColID = blockIdx.y; 

    outIndex = newColID * ldaTransform + newRowID; 

    if (newRowID < ldaTransform)
        ATransform[outIndex] = tile[threadIdx.x][threadIdx.y];
}

// only support the datatype of int8
__global__ void ColumnMajor2COl32Char4(int rows, int cols,
                                       const int8_t *A, int lda,
                                       int8_t *ATransform, int ldaTransform,
                                       int numThreadTile_except_lastBlock_X)
{
    // A is column major
    // each thread deals with 4 * 4 elements
    // In the column direction, use char4 to deal with the 4 in this direction 
    // In the row    direction, need to loop 4 times 
    int rowID = threadIdx.x + blockDim.x * blockIdx.x;
    int colID = threadIdx.y + blockDim.y * (blockIdx.y<<2);
    int x, y, inIndex, outIndex;

    // since we cast the pointer from char to char4 
    int   lda4 = lda>>2;
    char4 temp0 = {0,0,0,0};
    char4 temp1;
    char4 temp2;
    char4 temp3;
    char4 c4Array[4];

    const char4 *pInC4  = (const char4 *)A;
    char4       *pOutC4 = (      char4 *)ATransform;

    __shared__ char4 tile[COL32][BLOCK_DIMX + 1];


    if (rowID < lda4) 
    {
        for (int j=0; j<TILE_Y; j+=blockDim.y) 
        {
            x = rowID;
            y = colID + j;

            inIndex = x + y * lda4;

            if (y < cols)
            {
                tile[threadIdx.y + j][threadIdx.x] = pInC4[inIndex];
            }
            else
                tile[threadIdx.y + j][threadIdx.x] = temp0;
        }
    }

    __syncthreads();


    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // re-map all threads in each block for coalescing gmem write
    x = tid % blockDim.y;
    y = tid / blockDim.y;

    // we have re-mapped threads with task
    // the condition to determine boundary has changed
    // 8 is blockDim.y 
    if ( !numThreadTile_except_lastBlock_X || // rows%TILE_X == 0, then all blocks participate the work, otherwise
	     ((blockIdx.x != gridDim.x - 1) || (tid < restThreadTile_X * 8))) 
    {
        int baseNewRowID; 
        int partOutIndex; 

        temp0 = tile[x*4 + 0][y];
        temp1 = tile[x*4 + 1][y];
        temp2 = tile[x*4 + 2][y];
        temp3 = tile[x*4 + 3][y];

        // blockDim.y = 8,  left-shift 3 bits
        // blockDim.x = 32, left-shift 5 bits
        // baseNewRowID = x +  y * blockDim.x + blockIdx.x * 4 * blockDim.y * blockDim.x;
        baseNewRowID = x + (y<<5) + (blockIdx.x<<10);
        partOutIndex = blockIdx.y * (ldaTransform>>2) + baseNewRowID;

        // transpose the thread-tile (4x4)
        c4Array[0].x = temp0.x; c4Array[0].y = temp1.x; c4Array[0].z = temp2.x; c4Array[0].w = temp3.x;
        outIndex = partOutIndex + 0;
        pOutC4[outIndex] = c4Array[0];
        
        c4Array[1].x = temp0.y; c4Array[1].y = temp1.y; c4Array[1].z = temp2.y; c4Array[1].w = temp3.y;
        outIndex = partOutIndex + 8;
        pOutC4[outIndex] = c4Array[1];
        
        c4Array[2].x = temp0.z; c4Array[2].y = temp1.z; c4Array[2].z = temp2.z; c4Array[2].w = temp3.z;
        outIndex = partOutIndex + 16;
        pOutC4[outIndex] = c4Array[2];

        c4Array[3].x = temp0.w; c4Array[3].y = temp1.w; c4Array[3].z = temp2.w; c4Array[3].w = temp3.w;
        outIndex = partOutIndex + 24;
        pOutC4[outIndex] = c4Array[3];
    }
}

int transformColumnMajor2COl32(int rows, int cols,
                               const int8_t *A, int lda,
                               int8_t *ATransform, int ldaTransform)
{
    // A is stored in column-major
    // ATransform is stored in COL32 format
    if (A == NULL || ATransform == NULL)
    {
        printf("Error: input pointers are invaid\n");
        exit(-1);
    }

	if (rows%4 == 0 && cols%4 == 0)
	{
    	dim3 dimBlockC4;
    	dim3 dimGridC4;
		// total number of thread tiles in X
    	int  totThreadTile_X  =  rows / 4;	
		// rows / TILE_X: how many threadblock in X
		// each threadblock has 32 thread tile in X
    	int  numThreadTile_except_lastBlock_X = (rows / TILE_X) * 32; 
    	int  numThreadTile_valid_lastBlock_X = totThreadTile_X - numThreadTile_except_lastBlock_X;

    	dimBlockC4.x = TILE_X/4; // 32
    	dimBlockC4.y = TILE_Y/4; // 8

    	dimGridC4.x = (rows + TILE_X - 1) / TILE_X;
    	dimGridC4.y = (cols + TILE_Y - 1) / TILE_Y;


    	ColumnMajor2COl32Char4<<<dimGridC4, dimBlockC4>>>(rows, cols,
                                                    	  A, lda, 
                                                    	  ATransform, ldaTransform, 
                                                   		  numThreadTile_valid_lastBlock_X);
	}
	else
	{
    	dim3 dimBlock; 
    	dim3 dimGrid;

    	dimBlock.x = BLOCK_DIMX;
    	dimBlock.y = COL32; // must be 32 

    	dimGrid.x = (rows + dimBlock.x - 1) / dimBlock.x;
    	dimGrid.y = (cols + dimBlock.y - 1) / dimBlock.y;

    	ColumnMajor2COl32<<<dimGrid, dimBlock>>>(rows, cols,
                                                 A, lda, 
                                                 ATransform, ldaTransform);
	}

    return 0;
}

int transformCOl322ColumnMajor(int rows, int cols,
                               int *CTransform, int ldcTransform,
                               int *C,          int ldc)
{
    // input check
    // CTransform is stored in COL32 format
    // C is stored in column-major
    if (C == NULL || CTransform == NULL)
    {
        printf("Error: input pointers are invaid\n");
        exit(-1);
    }

    dim3 dimBlock; 
    dim3 dimGrid;

    dimBlock.x = COL32;// must be 32 
    dimBlock.y = COL32; 

    dimGrid.x  = (rows + dimBlock.y - 1) / dimBlock.y;
    dimGrid.y  = (cols + COL32      - 1) / COL32;

    COL322ColumnMajor<<<dimGrid, dimBlock>>>(rows, cols,
                                             CTransform, ldcTransform, 
                                             C, ldc);

    return 0;
}
