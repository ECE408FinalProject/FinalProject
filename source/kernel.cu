/* Possible Optimizations
 * 1. Shared memory for each thread block to store each block in the frame
 * 2. Constant memory for the entire frame for reference
 * 3. Use streams to simultaneously compute multiple n sizes and multiple frames
 */ 

 #define NUM_MODES		35
 #define NUM_CHANNELS	3
__constant__ float HERAMAND4[4][4] = {{1,1,1,1},
									  {1,-1,1,-1},
									  {1,1,-1,-1},
									  {1,-1,-1,1},
									 };

__constant__ float HERAMAND8[8][8] = {{1,1,1,1,1,1,1,1},
									  {1,-1,1,-1,1,-1,1,-1},
									  {1,1,-1,-1,1,1,-1,-1},
									  {1,-1,-1,1,1,-1,-1,1},
									  {1,1,1,1,-1,-1,-1,-1},
									  {1,-1,1,-1,-1,1,-1,1},
									  {1,1,-1,-1,-1,-1,1,1},
									  {1,-1,-1,1,-1,1,1,-1},
									 }
 __device__
void DCPrediction(ece408_frame *currentFrame, ece408_frame *predictionFrames) {

}


__device__
void SATD_luma(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_result * result){
	//compute unsorted SATD values
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int n = 0;
	if(blockDim.x == 2)
		n=4;
	else
		n=8;
	__shared__ int diffr[n][n];
	__shared__ int sum;
	for(int i = 0; i < 35; i++){
		//TODO needs to Handle 4 blocks if block size is 2 or 4
		sum = 0;
		if(blockDim.x == 2)
		{
			//handle a 4x4 block
			diffr[2*row][2*col] = currentFrame->y[2*row*currentFrame->width + 2*col] - predictionFrames[i].y[2*row*currentFrame->width + 2*col];
			diffr[2*row+1][2*col] = currentFrame->y[(2*row+1)*currentFrame->width + 2*col] - predictionFrames[i].y[(2*row+1)*currentFrame->width + 2*col];
			diffr[2*row][2*col+1] = currentFrame->y[2*row*currentFrame->width + 2*col+1] - predictionFrames[i].y[2*row*currentFrame->width + 2*col+1];
			diffr[2*row+1][2*col+1] = currentFrame->y[(2*row+1)*currentFrame->width + 2*col+1] - predictionFrames[i].y[(2*row+1)*currentFrame->width + 2*col+1];
			__syncthreads();
			int val = SATD(diffr, 2*row, 2*col);
			atomicAdd(&sum,abs(val));
			val = SATD(diffr, 2*row+1, 2*col);
			atomicAdd(&sum, abs(val));
			val = SATD(diffr, 2*row, 2*col+1);
			atomicAdd(&sum, abs(val));
			val = SATD(diffr, 2*row+1, 2*col+1);
			atomicAdd(&sum, abs(val));
		}
		else if(blockDim.x == 4)
		{
			//handle a 8x8 block
			diffr[2*row][2*col] = currentFrame->y[2*row*currentFrame->width + 2*col] - predictionFrames[i].y[2*row*currentFrame->width + 2*col];
			diffr[2*row+1][2*col] = currentFrame->y[(2*row+1)*currentFrame->width + 2*col] - predictionFrames[i].y[(2*row+1)*currentFrame->width + 2*col];
			diffr[2*row][2*col+1] = currentFrame->y[2*row*currentFrame->width + 2*col+1] - predictionFrames[i].y[2*row*currentFrame->width + 2*col+1];
			diffr[2*row+1][2*col+1] = currentFrame->y[(2*row+1)*currentFrame->width + 2*col+1] - predictionFrames[i].y[(2*row+1)*currentFrame->width + 2*col+1];
			__syncthreads();
			int val = SA8D(diffr, 2*row, 2*col);
			atomicAdd(&sum,abs(val));
			val = SA8D(diffr, 2*row+1, 2*col);
			atomicAdd(&sum, v);
			val = SA8D(diffr, 2*row, 2*col+1);
			atomicAdd(&sum, abs(val));
			val = SA8D(diffr, 2*row+1, 2*col+1);
			atomicAdd(&sum, abs(val));
			
		}
		else{
			//handle 1 8x8 block
			diffr[row][col] = currentFrame->y[row*currentFrame->width + col] - predictionFrames[i].y[row*currentFrame->width + col];
			int val = SA8D(diffr, row, col);
			atomicAdd(&sum,abs(val));
		}
		__syncthreads();
		if(threadIdx.x ==0 && threadIdx.y == 0)
			result->cr_satd_result[i] = (uint32_t) sum;
		
	}
}
__device__
int SATD(int diffr[][], int row, int col){
	int result = 0;
	for(int i = 0; i < 4; i++)
	{
		//multiply matrices
		result += (int)((float)diffr[row][i] * HERAMAND4[i][col]);
	}
	__syncthreads();
	//diffr[row][col] = result;
	return result;
}
__device__
int SA8D(int diffr[][], int row, int col){
	int result = 0;
	for(int i = 0; i < 8; i++)
	{
		//multiply matrices
		result += (int)((float)diffr[row][i] * HERAMAND8[i][col]);
	}
	__syncthreads();
	return result;
}
__device__
void SATD_Cr(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_result * result){
	//compute unsorted SATD values
	__shared__ int diffr[blockDim.x][blockDim.y];
	__shared__ int sum = 0;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int val;
	for(int i = 0; i < 35; i++){
		sum = 0;
		diffr[row][col] = currentFrame->cr[row*currentFrame->width + col] - predictionFrames[i].cr[row*currentFrame->width + col];
		__syncthreads();
		
		if(blockDim.x == 4){
			val = SATD(diffr,row,col);
		}
		else{
			val = SA8D(diffr,row,col);
		}
		//TODO atomic add the SATD values into result->cr_satd_result[i]
		atomicAdd(&sum, abs(val));
		__syncthreads();
		if(threadIdx.x ==0 && threadIdx.y == 0)
			result->cr_satd_result[i] = (uint32_t) sum;
	}
}
__device__
void SATD_Cb(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_result * result){
	//compute unsorted SATD values
	__shared__ int diffr[blockDim.x][blockDim.y];
	
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int val;
	for(int i = 0; i < 35; i++){
		sum = 0;
		diffr[row][col] = currentFrame->cr[row*currentFrame->width + col] - predictionFrames[i].cr[row*currentFrame->width + col];
		__syncthreads();
		
		if(blockDim.x == 4){
			val = SATD(diffr,row,col);
		}
		else{
			val = SA8D(diffr,row,col);
		}
		//TODO atomic add the SATD values into result->cb_satd_result[i]
		atomicAdd(&sum, abs(val));
		__syncthreads();
		if(threadIdx.x ==0 && threadIdx.y == 0)
			result->cb_satd_result[i] = (uint32_t) sum;
	}
}
__global__
void SATD(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_result * result){
	switch (blockIdx.z) {
		//blockDim.x == 2 implies luma size = 4 chroma size = N/A
		//TODO still need to be sorted
		case 0:
			SATD_luma(currentFrame, predictionFrames, result);
			break;

		case 1:
			if(blockDim.x !=2){
				SATD_Cr(currentFrame, predictionFrames, result);
			}
			else{
				result->cr_satd_results = null;
				result->cr_modes = null;
			}
			break;

		default:
			if(blockDim.x != 2){
				SATD_Cb(currentFrame, predictionFrames, result);
			}
			else[
				result->cb_satd_results = null;
				result->cb_modes = null;
			}
			break;
	}
}
__device__ 
void PlanarCb((ece408_frame *currentFrame, ece408_frame *predictionFrames) {
	int nTbS = blockDim.x;
	int col = blockDim.x*blockIdx.x + threadIdx.x; //col
	int row = blockDim.y*blockIdx.y + threadIdx.y; //row
	int height = currentFrame->height/2;
	int width = currentFrame->width/2;
	uint8_t top_b;
	uint8_t left_b;
	uint8_t top_x;
	uint8_t left_y;
	//skip partial blocks
	if(blockDim.x == gridDim.x-1){
		if(width%nTbS != 0)
			if(row<height && col < width)
				predictionFrame->cb[row*width+col] = currentFrame->cr[row*width+col];
		
	}
	if(blockDim.y == gridDim.y-1{
		if(height%nTbS != 0)
			if(row<height**col<width)
			predictionFrame->cb[row*width+col] = currentFrame->cr[row*width+col];
	}
	if(blockDim.x == 0 && blockDim.y == 0){
		if(blockDim.x != 4){
			top_b = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_b = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			top_x = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_y = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
		}
		else{
			top_b = (1 << 7);
			left_b = (1 << 7);
			top_x = (1 << 7);
			left_y = (1 << 7);
		}
	}
	else if(blockDim.x == 0){
		left_b = currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width];
		left_y = currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width];
		top_x = currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width + col];
		top_b = currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width + nTbS];
		if(blockDim.x != 4){
			left_b = (left_b + 2 * left_b + left_b + 2 ) >> 2;
			left_y = left_b;
			top_x = (currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width + col-1] + 2*top_x + currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width + col+1] + 2) >> 2;
			top_b = (currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width + nTbS-1] + 2*top_b + currentFrame->cb[((blockDim.y*blockIdx.y)-1)*width + nTbS+1] + 2) >> 2;
			
		}
	}
	else if(blockDim.y == 0){
		uint8_t top_pixels = currentFrame->cb[(blockDim.y*blockIdx.y)*width + (blockDim.x*blockIdx.x-1)];
		top_b = top_pixels;
		top_x = top_pixels;
		left_y = currentFrame->cb[(row*width)+(blockDim.x*blockIdx.x-1)];
		left_b = currentFrame->cb[(nTbS*width)+(blockDim.x*blockIdx.x-1)];
		
		
		if(blockDim.x != 4){
			//luma block size is not 4
			
			//filter
			top_pixels = (top_pixels + 2 * top_pixels + top_pixels + 2 ) >> 2;
			top_b = top_pixels;
			top_x = top_pixels;
			left_y = (currentFrame->cb[((row-1)*width)+(blockDim.x*blockIdx.x-1)] + 2*left_y + currentFrame->cb[(row+1*width)+(blockDim.x*blockIdx.x-1)]) >> 2;
			left_b = (currentFrame->cb[((nTbS-1)*width)+(blockDim.x*blockIdx.x-1)] + 2*left_b + currentFrame->cb[((nTbS+1)*width)+(blockDim.x*blockIdx.x-1)]) >> 2;
			
		}	
	}
	else{
		uint8_t top_b;
		uint8_t left_b;
		uint8_t top_x = currentFrame->cb[(blockDim.y*blockIdx.y)-1)*width + col];
		uint8_t left_y = currentFrame->cb[(row*width)+(blockDim.x*blockIdx.x-1)];
		
		if(blockIdx.x == gridDim.x-1){
		
			top_b = currentFrame->cb[(blockDim.y*blockIdx.y-1)*width + nTbS - 1];
			if(blockDim.x != 4){
				top_b = (top_b + 2*top_b + top_b) >> 2;
			}
		}
		else{
			top_b = currentFrame->cb[(blockDim.y*blockIdx.y-1)*width + nTbS];
			if(blockDim.x != 4){
				top_b = (currentFrame->cb[(blockDim.y*blockIdx.y-1)*width + nTbS - 1 ] + 2*top_b + currentFrame->[(blockDim.y*blockIdx.y-1)*width + nTbS] + 1) >> 2;
			}
		}
		if(blockIdx.y == gridDim.y-1){
			left_b = currentFrame->cb[(nTbS-1)*width + (blockDim.x*blockIdx.x-1)];
			if(blockDim.x != 4){
				left_b = (left_b + 2*left_b + left_b)>>2;
			}
		}
		else{
			left_b = currentFrame->cb[(nTbS)*width + (blockDim.x*blockIdx.x-1)];
			if(blockDim.x != 4){
				left_b = (currentFrame->cr[(nTbS-1)*width + (blockDim.x*blockIdx.x-1)] + 2*left_b + currentFrame->[(nTbS+1)*width + (blockDim.x*blockIdx.x-1)])>>2;
			}
		}
	}
	uint8_t result =( nTbS − 1 − col ) * left_y + ( col + 1 ) * top_b +
			( nTbS − 1 − row ) * top_x +
			( row + 1 ) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->cb[(row)*width + (col) = result;
}



__device__ 
void PlanarCr((ece408_frame *currentFrame, ece408_frame *predictionFrames) {
	int nTbS = blockDim.x;
	int col = blockDim.x*blockIdx.x + threadIdx.x; //col
	int row = blockDim.y*blockIdx.y + threadIdx.y; //row
	int height = currentFrame->height/2;
	int width = currentFrame->width/2;
	uint8_t top_b;
	uint8_t left_b;
	uint8_t top_x;
	uint8_t left_y;
	
	//skip partial blocks
	if(blockDim.x == gridDim.x-1){
		if(width%nTbS != 0)
			if(row<height && col < width)
				predictionFrame->cr[row*width+col] = currentFrame->cr[row*width+col];
		
	}
	if(blockDim.y == gridDim.y-1{
		if(height%nTbS != 0)
			if(row<height**col<width)
			predictionFrame->cr[row*width+col] = currentFrame->cr[row*width+col];
	}
	if(blockDim.x == 0 && blockDim.y == 0){
		if(blockDim.x != 4){
			top_b = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_b = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			top_x = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_y = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
		}
		else{
			top_b = (1 << 7);
			left_b = (1 << 7);
			top_x = (1 << 7);
			left_y = (1 << 7);
		}
	}
	else if(blockDim.x == 0){
		left_b = currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width];
		left_y = currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width];
		top_x = currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width + col];
		top_b = currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width + nTbS];
		if(blockDim.x != 4){
			left_b = (left_b + 2 * left_b + left_b + 2 ) >> 2;
			left_y = left_b;
			top_x = (currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width + col-1] + 2*top_x + currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width + col+1] + 2) >> 2;
			top_b = (currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width + nTbS-1] + 2*top_b + currentFrame->cr[((blockDim.y*blockIdx.y)-1)*width + nTbS+1] + 2) >> 2;
			
		}
	}
	else if(blockDim.y == 0){
		uint8_t top_pixels = currentFrame->cr[(blockDim.y*blockIdx.y)*width + (blockDim.x*blockIdx.x-1)];
		top_b = top_pixels;
		top_x = top_pixels;
		left_y = currentFrame->cr[(row*width)+(blockDim.x*blockIdx.x-1)];
		left_b = currentFrame->cr[(nTbS*width)+(blockDim.x*blockIdx.x-1)];
		
		
		if(blockDim.x != 4){
			//luma block size is not 4
			
			//filter
			top_pixels = (top_pixels + 2 * top_pixels + top_pixels + 2 ) >> 2;
			top_b = top_pixels;
			top_x = top_pixels;			
			left_y = (currentFrame->cr[((row-1)*width)+(blockDim.x*blockIdx.x-1)] + 2*left_y + currentFrame->cr[(row+1*width)+(blockDim.x*blockIdx.x-1)]) >> 2;
			left_b = (currentFrame->cr[((nTbS-1)*width)+(blockDim.x*blockIdx.x-1)] + 2*left_b + currentFrame->cr[((nTbS+1)*width)+(blockDim.x*blockIdx.x-1)]) >> 2;
			
		}	
	}
	else{
		uint8_t top_b;
		uint8_t left_b;
		uint8_t top_x = currentFrame->cr[(blockDim.y*blockIdx.y)-1)*width + col];
		uint8_t left_y = currentFrame->cr[(row*width)+(blockDim.x*blockIdx.x-1)];
		
		if(blockIdx.x == gridDim.x-1){
		
			top_b = currentFrame->cr[(blockDim.y*blockIdx.y-1)*width + nTbS - 1];
			if(blockDim.x != 4){
				top_b = (top_b + 2*top_b + top_b) >> 2;
			}
		}
		else{
			top_b = currentFrame->cr[(blockDim.y*blockIdx.y-1)*width + nTbS];
			if(blockDim.x != 4){
				top_b = (currentFrame->cr[(blockDim.y*blockIdx.y-1)*width + nTbS - 1 ] + 2*top_b + currentFrame->[(blockDim.y*blockIdx.y-1)*width + nTbS] + 1) >> 2;
			}
		}
		if(blockIdx.y == gridDim.y-1){
			left_b = currentFrame->cr[(nTbS-1)*width + (blockDim.x*blockIdx.x-1)];
			if(blockDim.x != 4){
				left_b = (left_b + 2*left_b + left_b)>>2;
			}
		}
		else{
			left_b = currentFrame->cb[(nTbS)*width + (blockDim.x*blockIdx.x-1)];
			if(blockDim.x != 4){
				left_b = (currentFrame->cr[(nTbS-1)*width + (blockDim.x*blockIdx.x-1)] + 2*left_b + currentFrame->[(nTbS+1)*width + (blockDim.x*blockIdx.x-1)])>>2;
			}
		}
	}
	uint8_t result =( nTbS − 1 − col ) * left_y + ( col + 1 ) * top_b +
			( nTbS − 1 − row ) * top_x +
			( row + 1 ) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->cr[(row)*width + (col) = result;
}

__device__
void PlanarLuma(ece408_frame *currentFrame, ece408_frame *predictionFrames) {
	int nTbS = 2*blockDim.x;
	int col = blockDim.x*blockIdx.x + threadIdx.x; //col
	int row = blockDim.y*blockIdx.y + threadIdx.y; //row
	int height = currentFrame->height;
	int width = currentFrame->width;
	uint8_t top_b;
	uint8_t left_b;
	uint8_t top_x;
	uint8_t top_x2;
	uint8_t left_y;
	uint8_t left_y2;
	//skip partial blocks
	if(blockDim.x == gridDim.x-1){
		if(width%nTbS != 0)
			if(2*row<height && 2*col < width)
				predictionFrame->cr[2*row*width+2*col] = currentFrame->cr[2*row*width+2*col];
			if(2*row+1<height && 2*col < width)
				predictionFrame->cr[(2*row+1)*width+2*col] = currentFrame->cr[(2*row+1)*width+2*col];
			if(2*row<height && 2*col+1 < width)
				predictionFrame->cr[2*row*width+2*col+1] = currentFrame->cr[2*row*width+2*col+1];
			if(2*row+1<height && 2*col +1< width)
				predictionFrame->cr[(2*row+1)*width+2*col] = currentFrame->cr[(2*row+1)*width+2*col+1];
		
	}
	if(blockDim.y == gridDim.y-1{
		if(height%nTbS != 0)
			if(2*row<height && 2*col < width)
				predictionFrame->cr[2*row*width+2*col] = currentFrame->cr[2*row*width+2*col];
			if(2*row+1<height && 2*col < width)
				predictionFrame->cr[(2*row+1)*width+2*col] = currentFrame->cr[(2*row+1)*width+2*col];
			if(2*row<height && 2*col+1 < width)
				predictionFrame->cr[2*row*width+2*col+1] = currentFrame->cr[2*row*width+2*col+1];
			if(2*row+1<height && 2*col +1< width)
				predictionFrame->cr[(2*row+1)*width+2*col] = currentFrame->cr[(2*row+1)*width+2*col+1];
	}
	//top left corner
	if(blockDim.x == 0 && blockDim.y == 0){
		//all reference pixels will be the same
		if(blockDim.x != 2){
			//luma block size not 4
			top_b = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_b = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			top_x = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			top_x2 = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_y = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			left_y2 = ((1 << 7) + 2 * (1 << 7) + (1 << 7) + 2 ) >> 2;
			
		}
		else{
			// value is not filtered if blockDim.x = 4;
			top_b = (1 << 7);
			left_b = (1 << 7);
			top_x = (1 << 7);
			top_x2 = (1 << 7);
			left_y = (1 << 7);
			left_y2 = (1 << 7);
			
		}
		
	}
	//left side
	else if(blockDim.x == 0){
		uint8_t left_pixels;
		uint8_t result;
		left_b = currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width];
		left_y = currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width];
		left_y2 = currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width];
		top_x = currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*col];
		top_b = currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*nTbS];
		top_x2 = currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + (2*col+1)];
		if(blockDim.x != 2){
			//luma block size is not 4
			
			//filter
			left_pixels = (left_b + 2 * left_b + left_b + 2 ) >> 2;
			left_b = left_pixels;
			left_y = left_pixels;
			left_y2 = left_pixels;
			top_x = (currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*col-1] + 2*top_x + currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*col+1] + 2) >> 2;
			top_b = (currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*nTbS-1] + 2*top_b + currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*nTbS+1] + 2) >> 2;
			top_x2 = (currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*col] + 2*top_x2 + currentFrame->y[((2*blockDim.y*blockIdx.y)-1)*width + 2*col+2] + 2) >> 2;
			
		}
	
	}
	//top side
	else if(blockDim.y == 0){
		
		uint8_t top_pixels = currentFrame->y[(2*blockDim.y*blockIdx.y)*width + (2*blockDim.x*blockIdx.x-1)];
		top_b = top_pixels;
		top_x = top_pixels;
		top_x2 = top_pixels;
		left_y = currentFrame->y[(2*row*width)+(2*blockDim.x*blockIdx.x-1)];
		left_b = currentFrame->y[(2*nTbS*width)+(2*blockDim.x*blockIdx.x-1)];
		left_y2 = currentFrame->y[((2*row+1)*width)+(2*blockDim.x*blockIdx.x-1)];
		
		if(blockDim.x != 2){
			//luma block size is not 4
			
			//filter
			top_pixels = (top_pixels + 2 * top_pixels + top_pixels + 2 ) >> 2;
			top_b = top_pixels;
			top_x = top_pixels;
			top_x2 = top_pixels;
			left_y = (currentFrame->y[((2*row-1)*width)+(2*blockDim.x*blockIdx.x-1)] + 2*left_y + currentFrame->y[(2*row+1*width)+(2*blockDim.x*blockIdx.x-1)]) >> 2;
			left_b = (currentFrame->y[((2*nTbS-1)*width)+(2*blockDim.x*blockIdx.x-1)] + 2*left_b + currentFrame->y[((2*nTbS+1)*width)+(vblockDim.x*blockIdx.x-1)]) >> 2;
			left_y2 = (currentFrame->y[((2*row)*width)+(2*blockDim.x*blockIdx.x-1)] + 2*left_y2 + currentFrame->y[((2*row+2)*width)+(2*blockDim.x*blockIdx.x-1)]) >> 2;
		}	
		
	}
	else{
	
		uint8_t top_b;
		uint8_t left_b;
		uint8_t top_x = currentFrame->y[(2*blockDim.y*blockIdx.y)-1)*width + 2*col];
		uint8_t top_x2 = currentFrame->y[(2*blockDim.y*blockIdx.y)-1)*width + 2*col+1];
		uint8_t left_y = currentFrame->y[(2*row*width)+(2*blockDim.x*blockIdx.x-1)];
		uint8_t left_y2 = currentFrame->y[((2*row+1)*width)+(2*blockDim.x*blockIdx.x-1)];
		if(blockIdx.x == dimGrid.x-1){
		
			top_b = currentFrame->y[(2*blockDim.y*blockIdx.y-1)*width + nTbS - 1];
			if(blockDim.x != 2){
			
				//block size is not 4 for luma gotta
				
				//filter
				top_b = (top_b + 2*top_b + top_b) >> 2;
			}
		}
		else{
			top_b = currentFrame->y[(2*blockDim.y*blockIdx.y-1)*width + nTbS];
			if(blockDim.x != 2){
			
				//block size is not 4 for luma gotta
				
				//filter
				top_b = (currentFrame->y[(2*blockDim.y*blockIdx.y-1)*width + nTbS - 1 ] + 2*top_b + currentFrame->y[(2*blockDim.y*blockIdx.y-1)*width + nTbS] + 1) >> 2;
			}
		}
		if(blockIdx.y == dimGrid.y-1){
			left_b = currentFrame->y[(nTbS-1)*width + (2*blockDim.x*blockIdx.x-1)];
			if(blockDim.x != 2){
				left_b = (left_b + 2*left_b + left_b)>>2;
			
			}
		}
		else{
			left_b = currentFrame->[(nTbS)*width + (2*blockDim.x*blockIdx.x-1)];
			if(blockDim.x != 2){
				left_b = (currentFrame->y[(nTbS-1)*width + (2*blockDim.x*blockIdx.x-1)] + 2*left_b + currentFrame->y[(nTbS+1)*width + (2*blockDim.x*blockIdx.x-1)])>>2;
			}
		}
	}
	
	result =( nTbS − 1 − col ) * left_y + ( col + 1 ) * top_b +
			( nTbS − 1 − row ) * top_x +
			( row + 1 ) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->y[(2*row)*width + (2*col) = result;
	result =( nTbS − 1 − col ) * left_y2 + ( col + 1 ) * top_b +
			( nTbS − 1 − row + 1 ) * top_x +
			( row+ 1 + 1 ) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->y[(2*row+1)*width + (2*col) = result;
	result =( nTbS − 1 − col + 1 ) * left_y + ( col + 1 + 1 ) * top_b +
			( nTbS − 1 − row ) * top_x2 +
			( row + 1 ) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->y[(2*row)*width + (2*col+1) = result;
	result =( nTbS − 1 − col ) * left_y2 + ( col + 1 ) * top_b +
			( nTbS − 1 − row + 1 ) * top_x2 +
			( row + 1 + 1) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->y[(2*row+1)*width + (2*col+1) = result;
	result =( nTbS − 1 − col +1 ) * left_y + ( col + 1 + 1) * top_b +
			( nTbS − 1 − row + 1 ) * top_x +
			( row + 1 + 1) * left_b + nTbS ) >> ( log2( nTbS ) + 1 );
	predictionFrames->y[(2*row+1)*width + (2*col+1) = result;
}
	
__device__
void PlanarPrediction(ece408_frame *currentFrame, ece408_frame *predictionFrames) {
	if(dimBlock.x != 2)
	{
		PlanarCr(currentFrame, predictionFrames);
		PlanarCb(currentFrame, predicitionFrames);
		
	}
	PlanarLuma(currentFrame, predictionFrames);
}

__host__
void AngularPredictionInitialize(ece408_frame *currentFrame, ece408_frame *predictionFrames, int iMode, int block_size) {
	
	bool modeHor = (dirMode < 18);
	bool modeVer = !modeHor;
	int intraPredAngle = modeVer ? iMode - VER_IDX : modeHor ? -(iMode - HOR_IDX) : 0;
	int absAng = abs(intraPredAngle);
	int signAng = intraPredAngle < 0 ? -1 : 1;

	int angTable[9]    = { 0,    2,    5,   9,  13,  17,  21,  26,  32 };
    int invAngTable[9] = { 0, 4096, 1638, 910, 630, 482, 390, 315, 256 }; // (256 * 32) / Angle
    int invAngle       = invAngTable[absAng];
    absAng             = angTable[absAng];
    intraPredAngle     = signAng * absAng;
}

__global__
void AngularPredictionLuma(ece408_frame *currentFrame, ece408_frame *predictionFrames, int iMode, int block_size) {

	extern __shared__ uint8_t *refAbove[];
	extern __shared__ uint8_t *refLeft[];

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int width = currentFrame->width;
	int height = currentFrame->height;

	uint8_t *refMain;
	uint8_t *refSide;

	if (threadIdx.y == 0) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = 128;
			refAbove[threadIdx.x + block_size*2] = 128;
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = 128;
			}
		}
		else if (blockIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = currentFrame->y[blockIdx.x * blockDim.x - 1 + width * row];
			refAbove[threadIdx.x + block_size*2] = currentFrame->y[blockIdx.x * blockDim.x - 1 + width * row];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->y[blockIdx.x * blockDim.x - 1 + width * row];
			}
		}
		else if (blockIdx.x == 0) {
			refAbove[threadIdx.x + block_size] = currentFrame->[col + width * (row - 1)];
			refAbove[threadIdx.x + block_size*2] = currentFrame->y[col + block_size + width * (row - 1)];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->y[col + width * (row - 1)];
			}
		}
		else if (blockIdx.y == gridDim.y - 1 && blockIdx.x == gridDim.x - 1) {
			if (width % block_size == 0 && height % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->y[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->y[width - 1 + width * row];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (height % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->y[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->y[col + block_size + width * (row - 1)];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.x == gridDim.x - 1 ) {
			if (width % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->y[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->y[width - 1 + width * (row - 1)];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
				}
			}
		}
		else {
			refAbove[threadIdx.x + block_size] = currentFrame->y[col + width * (row - 1)];
			refAbove[threadIdx.x + block_size*2] = currentFrame->y[col + block_size + width * (row - 1)];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
			}
		}
	}

	if (threadIdx.x == 0) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			refLeft[threadIdx.y + block_size] = 128;
			refLeft[threadidx.y + block_size*2] = 128;
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = 128;
			}
		}
		else if (blockIdx.y == 0) {
			refLeft[threadIdx.y + block_size] = currentFrame->y[col - 1 + width * row];
			refLeft[threadidx.y + block_size*2] = currentFrame->y[col - 1 + width * (row + block_size)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->y[col - 1 + width * row];
			}
		}
		else if (blockIdx.x == 0) {
			refLeft[threadIdx.y + block_size] = currentFrame->y[col + width * (blockIdx.y * blockDim.y - 1)];
			refLeft[threadidx.y + block_size*2] = currentFrame->y[col + width * (blockIdx.y * blockDim.y - 1)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->y[col + width * (blockIdx.y * blockDim.y - 1)];
			}
		}
		else if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
			if (width % block_size == 0 && height % block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->y[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->y[col + width * (height - block_size - 1)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (height % block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->y[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->y[col - 1 + width * (height - 1)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.x == gridDim.x - 1) {
			if (width & block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->y[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->y[col - 1 + width * (row + block_size)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->y[col - 1 + width * (row - 1)];
				}
			}
		}
		else {
			refLeft[threadIdx.y + block_size] = currentFrame->y[col - 1 + width * row];
			refLeft[threadIdx.y + block_size*2] = currentFrame->y[col - 1 + width * (row + block_size)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->y[col - 1 + width * (row -1)];
			}
		}
	}

	__syncthreads();

	refMain = modeVer ? &(refAbove[block_size]) : &(refLeft[block_size]);
	refSide = modeVer ? &(refLeft[block_size]) : &(refMain[block_size]);

	if (threadIdx.x == 0) {
		if (intraPredAngle < 0) {
			int invAngleSum = 128;
			for (int k = -1; k > blkSize * intraPredAngle >> 5; k--) {
				invAngleSum += invAngle;
				refMain[k] = refSide[invAngleSum >> 8];
			}
		}
	}
	__syncthreads();

	if (intraPredAngle == 0) {

	}
	else {
		
	}
}

/**
 * Main Intra Frame Prediction Function
 *
 * Function calls on the different modes of prediction based on the blockIdx.z variable.
 * 
 * @param currentFrame 		Pointer to the current frame being processed.
 * @param predictionFrames 	The array of frames that are predicted by the different modes
 */
__global__
void intra_frame_prediction(ece408_frame *currentFrame, ece408_frame *predictionFrames) {
	switch (blockIdx.z) {
		case 0:
			DCPrediction(currentFrame, predictionFrames);
			break;

		case 1:
			PlanarPrediction(currentFrame, predictionFrames);
			break;

		default:
			AngularPrediction(currentFrame, predictionFrames);
			break;
	}

}

/**
 * Main Function
 *
 * Function is responsible for managing all competition code. It will perform allocation
 * of device memory, call CUDA kernels, store results and free allocated memory.
 *
 * @param imgs			The array of frames to be processed.
 * @param num_frames	The number of frames in the above array.
 * @return	The ece408_intra_pred_result struct with all the results of the computation
 */
ece408_intra_pred_result *competition(ece408_frame *imgs, int num_frames) {
	ece408_intra_pred_result *results = new ece408_intra_pred_result results[num_frames];
	cudaError_t error;

	// Declare GPU memory to store one "real" frame  in the video
	ece408_frame *currentFrame_d;
	int frameSize = sizeof(ece408_frame);
	cudaMalloc((void**)&currentFrame, frameSize);
	int j = 0;
	// Declare GPU Memory for 35 temporary frames (1 for each intra prediction mode)
	ece408_frame *predictionFrames_d;
	cudaMalloc((void**)&predictionFrames), num_frames * frameSize);


	// For each frame, call the kernels	
	for (int i = 0; i < num_frames; i++) {
		// Note: We may need to insert another for loop here to loop different n

		ece408_frame currentFrame_h = imgs[i];

		// Set kernel dimensions
		//need to do this once for n = 2 n = 4 n = 8 n = 16
		for(int n = 2; n < 17; n *= 2, j++){
			dim3 predictionDimGrid((n-1)/(currentFrame_h.width/2)+1, (n-1)/(currentFrame_h.height/2)+1, NUM_MODES);
			dim3 predictionDimBlock(n, n, NUM_CHANNELS);
			
			
			//this is different. 
			if(n == 2){ //dont compute Cr and Cb
				dim3 satdDimGrid((2-1)/(currentFrame_h.width/2)+1, (2-1)/(currentFrame_h.height/2)+1, NUM_CHANNELS);
				dim3 satdDimBlock(2, 2, NUM_MODES);
			}
			else if(n == 4){ //size 4 blocks for cr/cb and 8 for y
				dim3 satdDimGrid((4-1)/(currentFrame_h.width/2)+1, (4-1)/(currentFrame_h.height/2)+1, NUM_CHANNELS);
				dim3 satdDimBlock(4, 4, NUM_MODES);
			}
			else{//size 8 for both
				dim3 satdDimGrid((8-1)/(currentFrame_h.width/2)+1, (8-1)/(currentFrame_h.height/2)+1, NUM_CHANNELS);
				dim3 satdDimBlock(8, 8, NUM_MODES);
			}

			// Copy the current frame into the GPU memory (try to optimize with constant memory)
			error = cudaMemcpy(currentFrame_d, currentFrame_h, frameSize, cudaMemcpyHostToDevice);
			if (error != cudaSuccess) FATAL("Unable to copy memory");

			// Call kernel to populate all 35 frames with the 35 addressing modes
			intra_frame_prediction<<<predictionDimGrid, predictionDimBlock>>>(currentFrame, predictionFrames);
			error = cudaDeviceSynchronize();
			if (error != cudaSuccess) FATAL("Unable to launch/execute kernel");

			// Call kernel to perform SATD on each block within the frame
			SATD(currentFrame, predictionFrames, &result[j]);
			// Copy result into the results array
		}
	}

	// Free allocated memory

	return results;
}
