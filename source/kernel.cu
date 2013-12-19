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
	int dc_val_y = 0;
	int dc_val_cb = 0;
	int dc_val_cr = 0;
	
	//blockDim corresponds to chroma dimension
	int col_c = blockIdx.x*blockDim.x+threadIdx.x;//current column and row of chroma frame
	int row_c = blockIdx.y*blockDim.y+threadIdx.y;
	int col_l = col_c*2;//y offset because twice as large as chroma
	int row_l = row_c*2;
	int cell_c = row_c*currentFrame->width/2+col_c;//current pixel in chroma frame
	int prevrow_firstcell = (blockIdx.y*blockDim.y*2 - 1)*currentFrame->width + blockIdx.x*blockDim.x*2;//previous row but first column of current block
	int prevcol_firstcell = blockIdx.y*blockDim.y*2*currentFrame->width + (blockIdx.x - 1)*blockDim.x*2 + blockDim.x*2 - 1;//previous column but first row of current block
	
	int cell_y0 = row_l*currentFrame->width+col_l;//grabs 4 luma pixels per thread since luma block is 4 times the size as a chroma block
	int cell_y1 = row_l*currentFrame->width+(col_l+1);
	int cell_y2 = (row_l+1)*currentFrame->width+col_l;
	int cell_y3 = (row_l+1)*currentFrame->width+(col_l+1);

	//rightmost column, partial blocks
	if(blockIdx.x == gridDim.x-1 && gridDim.x % blockDim.x != 0)
	{
		if(col_c < currentFrame->width/2 && row_c < currentFrame->height/2)
		{
			if(blockDim.x != 2){
				predictionFrames[1].cb[cell_c] = currentFrame[1].cb[cell_c];
				predictionFrames[1].cr[cell_c] = currentFrame[1].cr[cell_c];
			}
			else{
				predictionFrames[1].cb = NULL;
				predictionFrames[1].cr = NULL;
			}
			predictionFrames[1].y[cell_y0] = currentFrame[1].y[cell_y0];
			predictionFrames[1].y[cell_y1] = currentFrame[1].y[cell_y1];
			predictionFrames[1].y[cell_y2] = currentFrame[1].y[cell_y2];
			predictionFrames[1].y[cell_y3] = currentFrame[1].y[cell_y3];
		}
	}
	//bottom row, partial blocks
	if(blockIdx.y == gridDim.y-1 && gridDim.y % blockDim.y != 0)
	{
		if(row_c < currentFrame->height/2 && col_c < currentFrame->width/2)
		{
			if(blockDim.x != 2){
				predictionFrames[1].cb[cell_c] = currentFrame[1].[cell_c];
				predictionFrames[1].cr[cell_c] = currentFrame[1].[cell_c];
			}
			else{
				predictionFrames[1].cb = NULL;
				predictionFrames[1].cr = NULL;
			}
			predictionFrames[1].y[cell_y0] = currentFrame[1].y[cell_y0];
			predictionFrames[1].y[cell_y1] = currentFrame[1].y[cell_y1];
			predictionFrames[1].y[cell_y2] = currentFrame[1].y[cell_y2];
			predictionFrames[1].y[cell_y3] = currentFrame[1].y[cell_y3];
		}
	}
	// if on first block (top left block, no values to base predicted on)
	if(blockIdx.x - 1 < 0 && blockIdx.y - 1 < 0)
	{
		dc_val_y = 128;
		if(blockDim.x != 2){
			dc_val_cb = 128;
			dc_val_cr = 128;
		}
	}
	//first column blocks
	else if(blockIdx.x - 1 < 0)
	{
		// add values to the left
		for(int i = 0; i < blockDim.y*2; i++)
		{
			if(threadIdx.x == 0 && threadIdx.y == 0){
				//add all of left column and replace top row with first of left column
				int prev_col_y = (blockIdx.y*blockDim.y*2 + i)*currentFrame->width + (blockIdx.x - 1)*blockDim.x*2 + blockDim.x*2 - 1;
				int first_col_y = (blockIdx.y*blockDim.y*2)*currentFrame->width + (blockIdx.x - 1)*blockDim.x*2 + blockDim.x*2 - 1;
				int prev_col_c = (blockIdx.y*blockDim.y + i/2)*currentFrame->width/2 + (blockIdx.x - 1)*blockDim.x + blockDim.x - 1;
				int first_col_c = (blockIdx.y*blockDim.y)*currentFrame->width/2 + (block.x - 1)*blockDim.x + blockDim.x - 1;

				dc_val_y += currentFrame->y[prev_col_y] + currentFrame->y[first_col_y];
				if(blockDim.x != 2){
					dc_val_cb += currentFrame->cb[prev_col_c] + currentFrame->cb[first_col_c];
					dc_val_cr += currentFrame->cr[prev_col_c] + currentFrame->cr[first_col_c];			
				}
			}
		}
		//average the dc values
		dc_val_y /= (4*blockDim.y);
		dc_val_cb /= (2*blockDim.y);
		dc_val_cr /= (2*blockDim.y);
	}
	//top row blocks
	else if(blockIdx.y - 1 < 0)
	{
		// add values above
		for(int i = 0; i < blockDim.x*2; i++)
		{
			if(threadIdx.x == 0 && threadIdx.y == 0){
				int prev_row_y = (blockIdx.y*blockDim.y*2 - 1)*currentFrame->width + blockIdx.x*blockDim.x*2 + i;
				int first_row_y = (blockIdx.y*blockDim.y*2 - 1)*currentFrame->width + blockIdx.x*blockDim.x*2;
				int prev_row_c = (blockIdx.y*blockDim.y - 1)*currentFrame->width/2 + blockIdx.x*blockDim.x + i/2;
				int first_row_c = (blockIdx.y*blockDim.y - 1)*currentFrame->width/2 + blockIdx.x*blockDim.x;

				dc_val_y += currentFrame->y[prev_row_y] + currentFrame->y[prev_row_y];
				if(blockDim.x != 2){
					dc_val_cb += currentFrame->cb[prev_row_c] + currentFrame->cb[first_row_c];
					dc_val_cr += currentFrame->cr[prev_row_c] + currentFrame->cr[first_row_c];
				}
			}
		}
		//average results
		dc_val_y /=(4*blockDim.x);
		dc_val_cb /= (2*blockDim.x);
		dc_val_cr /= (2*blockDim.x);
	}
	//middle full blocks
	else
	{
		for(int i = 0; i < blockDim.x*2; i++)
		{
			if(threadIdx.x == 0 && threadIdx.y = 0){
				int prev_row_y = (blockIdx.y*blockDim.y*2 - 1)*currentFrame->width + blockIdx.x*blockDim.x*2 + i;
				int prev_col_y = (blockIdx.y*blockDim.y*2 + i)*currentFrame->width + (blockIdx.x - 1)*blockDim.x*2 + blockDim.x*2 - 1;
				int prev_row_c = (blockIdx.y*blockDim.y - 1)*currentFrame->width/2 + blockIdx.x*blockDim.x + i/2;
				int prev_col_c = (blockIdx.y*blockDim.y + i/2)*currentFrame->width/2 + (blockIdx.x - 1)*blockDim.x + blockDim.x - 1;

				dc_val_y += currentFrame->y[prev_col_y] + currentFrame->y[prev_row_y];
				if(blockDim.x != 2){
					dc_val_cb += currentFrame->cb[prev_col_c] + currentFrame->cb[prev_row_c];
					dc_val_cr += currentFrame->cr[prev_col_c] + currentFrame->cr[prev_row_c];
				}
			}
		}
		//average results
		dc_val_y /= (4*blockDim.x);
		dc_val_cb /= (2*blockDim.x);
		dc_val_cr /= (2*blockDim.x);
	}
	if(blockDim.x != 2){
		predictionFrames[1].cb[cell_c] = (uint8_t)dc_val_cb;
		predictionFrames[1].cr[cell_c] = (uint8_t)dc_val_cr;
	}
	else{
		predictionFrames[1].cb[cell_c] = NULL;
		predictionFrames[1].cr[cell_c] = NULL;
	}
	predictionFrames[1].y[cell_y0] = (uint8_t)dc_val_y;
	predictionFrames[1].y[cell_y1] = (uint8_t)dc_val_y;
	predictionFrames[1].y[cell_y2] = (uint8_t)dc_val_y;
	predictionFrames[1].y[cell_y3] = (uint8_t)dc_val_y;
	//stated in spec
	if(blockDim.x*2 < 32){
		//if not in first column or row
		if(blockIdx.x > 0 && blockIdx.y > 0){
			//if we are in last partial column or in last partial row do nothing
			if((blockIdx.x == gridDim.x-1 && gridDim.x % blockDim.x != 0) || (blockIdx.y == gridDim.y-1 && gridDim.y % blockDim.y != 0)){}
			else{
				//replace top corner luma of block as stated in spec
				predictionFrames[1].y[0] = (currentFrame->y[prevcol_firstcell] + 2*dc_val_y + currentFrame->y[prevrow_firstcell] + 2)>>2;
				//replace top row and left column as stated in spec
				for(int i = 1; i < blockDim.x*2; i++){
					int prev_row_y = (blockIdx.y*blockDim.y*2 - 1)*currentFrame->width + blockIdx.x*blockDim.x*2 + i;
					int curr_row_y = (blockIdx.y*blockDim.y*2)*curentFrame->width + blockIdx.x*blockDim.x*2 + i;
					predictionFrames[1].y[curr_row_y] = (currentFrame[prev_row_y] + 3*(uint8_t)dc_val_y + 2)>>2;
					int prev_col_y = (blockIdx.y*blockDim.y*2 + i)*currentFrame->width + (blockIdx.x - 1)*blockDim.x*2 + blockDim.x*2 - 1;
					int curr_col_y = (blockIdx.y*blockDim.y*2 + i)*currentFrame->width + blockIdx.x*blockDim.x*2;
					predictionFrames[1].y[curr_col_y] = (currentFrame[prev_col_y] + 3*(uint8_t)dc_val_y + 2)>>2;
				}
			}
		}
		//if in first block
		else if(blockIdx.x == 0 && blockIdx.y == 0){
			predictionFrames[1].y[0] = (128 + 2*dc_val_y + 128 + 2)>>2;
			//replace first row and column of values
			for(int i = 1; i < blockDim.x*2; i++){
				predictionFrames[1].y[curr_row_y] = (128 + 3*(uint8_t)dc_val_y + 2)>>2;
				predictionFrames[1].y[curr_col_y] = (128 + 3*(uint8_t)dc_val_y + 2)>>2;
			}
		}
		//if in first column and not in last partial block of column
		else if (blockIdx.x == 0 && !(blockIdx.y == gridDim.y-1 && gridDim.y % blockDim.y != 0)){
			predictionFrames[1].y[0] = (128 + 2*dc_val_y + currentFrame->y[prevrow_firstcell] + 2)>>2;
			for(int i = 1; i < blockDim.x*2; i++){
				int prev_row_y = (blockIdx.y*blockDim.y*2 - 1)*currentFrame->width + blockIdx.x*blockDim.x*2 + i;
				int curr_row_y = (blockIdx.y*blockDim.y*2)*curentFrame->width + blockIdx.x*blockDim.x*2 + i;
				predictionFrames[1].y[curr_row_y] = (currentFrame[prev_row_y] + 3*(uint8_t)dc_val_y + 2)>>2;
			}
		}
		//if in first row and not in last partial block of row
		else if (blockIdx.y == 0 && !(blockIdx.x == gridDim.x-1 && gridDim.x % blockDim.x != 0)){
			predictionFrames[1].y[0] = (currentFrame->y[prevcol_firstcell] + 2*dc_val_y + 128 + 2)>>2;
			for(int i = 1; i < blockDim.x*2; i++){
				int prev_col_y = (blockIdx.y*blockDim.y*2 + i)*currentFrame->width + (blockIdx.x - 1)*blockDim.x*2 + blockDim.x*2 - 1;
				int curr_col_y = (blockIdx.y*blockDim.y*2 + i)*currentFrame->width + blockIdx.x*blockDim.x*2;
				predictionFrames[1].y[curr_col_y] = (currentFrame[prev_col_y] + 3*(uint8_t)dc_val_y + 2)>>2;
			}
		}
	}
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
void SATD_Cr(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_intra_pred_result * result){
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
void SATD_Cb(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_intra_pred_result * result){
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
		if(threadIdx.x == 0 && threadIdx.y == 0)
			result->cb_satd_result[i] = (uint32_t) sum;
	}
}
__global__
void SATD(ece408_frame *currentFrame, ece408_frame *predictionFrames, ece408_intra_pred_result * result){
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
				result->cr_satd_results = NULL;
				result->cr_modes = NULL;
			}
			break;

		default:
			if(blockDim.x != 2){
				SATD_Cb(currentFrame, predictionFrames, result);
			}
			else{
				result->cb_satd_results = NULL;
				result->cb_modes = NULL;
			}
			break;
	}
	//reorder the SATD mode indicies so that the best results appear first in the corresponding modes array
	uint8_t *tempy_mode = NULL;
	uint8_t *tempcb_mode = NULL;
	uint8_t *tempcr_mode = NULL;
	if(threadIdx.x == 0 && threadIdx.y == 0){
		int num_unsorted = 35;
		for(int i = 1; i < num_unsorted; i++){
			if(result->y_modes[i-1] > result->y_modes[i] && result->y_modes != NULL){
				tempy_mode = result->y_modes[i-1];
				result->y_modes[i-1] = result->y_modes[i];
				result->y_modes[i] = tempy_mode;
			}
			if(result->cb_modes[i-1] > result->cb_modes[i] && result->cb_modes != NULL){
				tempcb_mode = result->cb_modes[i-1];
				result->cb_modes[i-1] = result->cb_modes[i];
				result->cb_modes[i] = tempcb_mode;
			}
			if(result->cr_modes[i-1] > result->cr_modes[i] && result->cr_modes != NULL){
				tempcr_mode = result->cr_modes[i-1];
				result->cr_modes[i-1] = result->cr_modes[i];
				result->cr_modes[i] = tempcr_mode;
			}
			num_unsorted--;
		}
	}
}
__device__ 
void PlanarCb(ece408_frame *currentFrame, ece408_frame *predictionFrames) {
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
	if(blockIdx.x == gridDim.x-1){
		if(width%nTbS != 0)
			if(row<height && col < width)
				predictionFrame->cb[row*width+col] = currentFrame->cr[row*width+col];
		
	}
	if(blockIdx.y == gridDim.y-1){
		if(height%nTbS != 0){
			if(row<height**col<width){
			predictionFrame->cb[row*width+col] = currentFrame->cr[row*width+col];
			}
		}	
	}
	if(blockIdx.x == 0 && blockIdx.y == 0){
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
	else if(blockIdx.x == 0){
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
	else if(blockIdx.y == 0){
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
	if(blockIdx.x == gridDim.x-1){
		if(width%nTbS != 0)
			if(row<height && col < width)
				predictionFrame->cr[row*width+col] = currentFrame->cr[row*width+col];
		
	}
	if(blockIdx.y == gridDim.y-1{
		if(height%nTbS != 0)
			if(row<height**col<width)
			predictionFrame->cr[row*width+col] = currentFrame->cr[row*width+col];
	}
	if(blockIdx.x == 0 && blockIdx.y == 0){
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
	else if(blockIdx.x == 0){
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
	else if(blockIdx.y == 0){
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
	if(blockIdx.x == gridDim.x-1){
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
	if(blockIdx.y == gridDim.y-1{
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
	if(blockIdx.x == 0 && blockIdx.y == 0){
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
	else if(blockIdx.y == 0){
		
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
		PlanarCb(currentFrame, predictionFrames);
		
	}
	PlanarLuma(currentFrame, predictionFrames);
}

__host__
void AngularPredictionInitialize(ece408_frame *currentFrame_h, ece408_frame *predictionFrames_d, int iMode, int block_size) {

	// Use array to determine whether to filter reference arrays
	unsigned char IntraFilterType[][35] =
	{
    /*  4x4  */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    /*  8x8  */ { 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
    /* 16x16 */ { 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
    /* 32x32 */ { 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1 },
    /* 64x64 */ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	};
	
	// Set up prediction angle data (same as reference code)
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

    dim3 lumaDimGrid((currentFrame_h->width + block_size - 1)/block_size, (currentFrame_h->height + block_size - 1)/block_size, 1);
    dim3 lumaDimBlock(block_size, block_size, 1);

    dim3 chromaDimGrid((currentFrame_h->width/2 + block_size/2 - 1)/(block_size/2), (currentFrame_h->height/2 + block_size/2 - 1)/(block_size/2), 1);
    dim3 chromaDimBlock(block_size/2, block_size/2, 1);

    // Copy the frame to memory
    ece408_frame *currentFrame_d;
	int frameSize = sizeof(ece408_frame);
	cudaMalloc((void**)&currentFrame_d, frameSize);

	error = cudaMemcpy(currentFrame_d, currentFrame_h, frameSize, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) FATAL("Unable to copy memory");

	// Call Luma channel kernel
    AngularPredictionLuma<<<lumaDimGrid, lumaDimBlock, 2 * (sizeof(uint8_t) * block_size * 3)>>>
    	(currentFrame_d, predictionFrames_d, iMode, block_size, intraPredAngle, modeHor, modeVer, (bool)IntraFilterType[block_size][iMode]);
    error = cudaDeviceSynchronize();
	if (error != cudaSuccess) FATAL("Unable to launch/execute kernel");

	// Call ChromaB channel kernel
    AngularPredictionCB<<<chromaDimGrid, chromaDimBlock, 2 * (sizeof(uint8_t) * block_size * 3>>>
    	(currentFrame_d, predictionFrames_d, iMode, block_size, intraPredAngle, modeHor, modeVer, (bool)IntraFilterType[block_size][iMode]);
   	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) FATAL("Unable to launch/execute kernel");

	// Call ChromaR channel kernel
    AngularPredictionCR<<<chromaDimGrid, chromaDimBlock, 2 * (sizeof(uint8_t) * block_size * 3>>>
    	(currentFrame_d, predictionFrames_d, iMode, block_size, intraPredAngle, modeHor, modeVer, (bool)IntraFilterType[block_size][iMode]);
    error = cudaDeviceSynchronize();
	if (error != cudaSuccess) FATAL("Unable to launch/execute kernel");
}

__global__
void AngularPredictionLuma(ece408_frame *currentFrame, ece408_frame *predictionFrames, int iMode, int block_size, int intraPredAngle, bool modeHor, bool modeVer, bool filterFlag) {

	extern __shared__ uint8_t *shared[];
	uint8_t *refAbove = shared;
	uint8_t *refLeft = &refAbove[block_size * 3];

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int width = currentFrame->width;
	int height = currentFrame->height;

	uint8_t *refMain;
	uint8_t *refSide;

	// Initialize top reference array
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

	// Initialize left reference array
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

	// Filter if required
	if (filterFlag) {
		if (threadIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = (refAbove[threadIdx.x - 1 + block_size] + refAbove[threadIdx.x + block_size] * 2 + refAbove[threadIdx.x + 1 + block_size]) >> 2;
			if (threadIdx.x != block_size - 1) {
				refAbove[threadIdx.x * 2 + block_size] = (refAbove[threadIdx.x * 2 - 1 + block_size] + refAbove[threadIdx.x * 2 + block_size] * 2 + refAbove[threadIdx * 2 + 1 + block_size]) >> 2;
			}
		}
		if (threadIdx.x == 0) {
			refAbove[threadIdx.y + block_size] = (refAbove[threadIdx.y - 1 + block_size] + refAbove[threadIdx.y + block_size] * 2 + refAbove[threadIdx.y + 1 + block_size]) >> 2;
			if (threadIdx.y != block_size - 1) {
				refAbove[threadIdx.y * 2 + block_size] = (refAbove[threadIdx.y * 2 - 1 + block_size] + refAbove[threadIdx.y * 2 + block_size] * 2 + refAbove[threadIdx.y * 2 + 1 + block_size]) >> 2;
			}
		}
	}

	__syncthreads();

	// Set main and side reference arrays
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

	// Populate the prediction frames with data from the reference arrays
	if (intraPredAngle == 0) {
		predictionFrames[iMode].y[col + row * width] = refMain[threadIdx.x + 1];
	}
	else {
		int deltaPos = (threadIdx.y + 1) * intraPredAngle;
		int deltaInt = deltaPos >> 5;
		int deltaFract = deltaPos & 31;
		
		if (deltaFract) {
			refMainIndex = threadIdx.x + deltaInt + 1;
			predictionFrames[iMode].y[col + row * width] = ((32 - deltaFract) * refMain[refMainIndex] + deltaFract * refMain[refMainIndex + 1] + 16) >> 5;
		}
		else {
			predictionFrames[iMode].y[col + row * width] = refMain[threadIdx.x + deltaInt + 1];
		}
	}

	// Flip if horizontal mode
	if (modeHor) {
		uint8_t tmp;
		tmp = predictionFrame[iMode].y[col + row * width];
		__syncthreads();
		predictionFrame[iMode].y[(blockIdx.x * blockDim.x + threadIdx.y) + (blockIdx.y * blockDim.y + threadIdx.x) * width] = tmp;
	}
}

__global__
void AngularPredictionCB(ece408_frame *currentFrame, ece408_frame *predictionFrames, int iMode, int block_size, int intraPredAngle, bool modeHor, bool modeVer, bool filterFlag) {

	extern __shared__ uint8_t *shared[];
	uint8_t *refAbove = shared;
	uint8_t *refLeft = &refAbove[block_size * 3];

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int width = currentFrame->width;
	int height = currentFrame->height;

	uint8_t *refMain;
	uint8_t *refSide;

	// Initialize top reference array
	if (threadIdx.y == 0) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = 128;
			refAbove[threadIdx.x + block_size*2] = 128;
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = 128;
			}
		}
		else if (blockIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = currentFrame->cb[blockIdx.x * blockDim.x - 1 + width * row];
			refAbove[threadIdx.x + block_size*2] = currentFrame->cb[blockIdx.x * blockDim.x - 1 + width * row];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->cb[blockIdx.x * blockDim.x - 1 + width * row];
			}
		}
		else if (blockIdx.x == 0) {
			refAbove[threadIdx.x + block_size] = currentFrame->[col + width * (row - 1)];
			refAbove[threadIdx.x + block_size*2] = currentFrame->cb[col + block_size + width * (row - 1)];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->cb[col + width * (row - 1)];
			}
		}
		else if (blockIdx.y == gridDim.y - 1 && blockIdx.x == gridDim.x - 1) {
			if (width % block_size == 0 && height % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->cb[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->cb[width - 1 + width * row];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (height % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->cb[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->cb[col + block_size + width * (row - 1)];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.x == gridDim.x - 1 ) {
			if (width % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->cb[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->cb[width - 1 + width * (row - 1)];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
				}
			}
		}
		else {
			refAbove[threadIdx.x + block_size] = currentFrame->cb[col + width * (row - 1)];
			refAbove[threadIdx.x + block_size*2] = currentFrame->cb[col + block_size + width * (row - 1)];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
			}
		}
	}

	// Initialize left refence array
	if (threadIdx.x == 0) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			refLeft[threadIdx.y + block_size] = 128;
			refLeft[threadidx.y + block_size*2] = 128;
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = 128;
			}
		}
		else if (blockIdx.y == 0) {
			refLeft[threadIdx.y + block_size] = currentFrame->cb[col - 1 + width * row];
			refLeft[threadidx.y + block_size*2] = currentFrame->cb[col - 1 + width * (row + block_size)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->cb[col - 1 + width * row];
			}
		}
		else if (blockIdx.x == 0) {
			refLeft[threadIdx.y + block_size] = currentFrame->cb[col + width * (blockIdx.y * blockDim.y - 1)];
			refLeft[threadidx.y + block_size*2] = currentFrame->cb[col + width * (blockIdx.y * blockDim.y - 1)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->cb[col + width * (blockIdx.y * blockDim.y - 1)];
			}
		}
		else if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
			if (width % block_size == 0 && height % block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->cb[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->cb[col + width * (height - block_size - 1)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (height % block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->cb[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->cb[col - 1 + width * (height - 1)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.x == gridDim.x - 1) {
			if (width & block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->cb[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->cb[col - 1 + width * (row + block_size)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->cb[col - 1 + width * (row - 1)];
				}
			}
		}
		else {
			refLeft[threadIdx.y + block_size] = currentFrame->cb[col - 1 + width * row];
			refLeft[threadIdx.y + block_size*2] = currentFrame->cb[col - 1 + width * (row + block_size)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->cb[col - 1 + width * (row -1)];
			}
		}
	}

	__syncthreads();

	// Filter if required
	if (filterFlag) {
		if (threadIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = (refAbove[threadIdx.x - 1 + block_size] + refAbove[threadIdx.x + block_size] * 2 + refAbove[threadIdx.x + 1 + block_size]) >> 2;
			if (threadIdx.x != block_size - 1) {
				refAbove[threadIdx.x * 2 + block_size] = (refAbove[threadIdx.x * 2 - 1 + block_size] + refAbove[threadIdx.x * 2 + block_size] * 2 + refAbove[threadIdx * 2 + 1 + block_size]) >> 2;
			}
		}
		if (threadIdx.x == 0) {
			refAbove[threadIdx.y + block_size] = (refAbove[threadIdx.y - 1 + block_size] + refAbove[threadIdx.y + block_size] * 2 + refAbove[threadIdx.y + 1 + block_size]) >> 2;
			if (threadIdx.y != block_size - 1) {
				refAbove[threadIdx.y * 2 + block_size] = (refAbove[threadIdx.y * 2 - 1 + block_size] + refAbove[threadIdx.y * 2 + block_size] * 2 + refAbove[threadIdx.y * 2 + 1 + block_size]) >> 2;
			}
		}
	}

	__syncthreads();

	// Set Main and side reference arrays
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

	// Populate the prediction frames with data from the reference arrays
	if (intraPredAngle == 0) {
		predictionFrames[iMode].cb[col + row * width] = refMain[threadIdx.x + 1];
	}
	else {
		int deltaPos = (threadIdx.y + 1) * intraPredAngle;
		int deltaInt = deltaPos >> 5;
		int deltaFract = deltaPos & 31;
		
		if (deltaFract) {
			refMainIndex = threadIdx.x + deltaInt + 1;
			predictionFrames[iMode].cb[col + row * width] = ((32 - deltaFract) * refMain[refMainIndex] + deltaFract * refMain[refMainIndex + 1] + 16) >> 5;
		}
		else {
			predictionFrames[iMode].cb[col + row * width] = refMain[threadIdx.x + deltaInt + 1];
		}
	}

	// Flip if horizontal mode
	if (modeHor) {
		uint8_t tmp;
		tmp = predictionFrame[iMode].cb[col + row * width];
		__syncthreads();
		predictionFrame[iMode].cb[(blockIdx.x * blockDim.x + threadIdx.y) + (blockIdx.y * blockDim.y + threadIdx.x) * width] = tmp;
	}
}

__global__
void AngularPredictionCR(ece408_frame *currentFrame, ece408_frame *predictionFrames, int iMode, int block_size, int intraPredAngle, bool modeHor, bool modeVer, bool filterFlag) {

	extern __shared__ uint8_t *shared[];
	uint8_t *refAbove = shared;
	uint8_t *refLeft = &refAbove[block_size * 3];

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int width = currentFrame->width;
	int height = currentFrame->height;

	uint8_t *refMain;
	uint8_t *refSide;

	// Initialize top reference array
	if (threadIdx.y == 0) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = 128;
			refAbove[threadIdx.x + block_size*2] = 128;
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = 128;
			}
		}
		else if (blockIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = currentFrame->cr[blockIdx.x * blockDim.x - 1 + width * row];
			refAbove[threadIdx.x + block_size*2] = currentFrame->cr[blockIdx.x * blockDim.x - 1 + width * row];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->cr[blockIdx.x * blockDim.x - 1 + width * row];
			}
		}
		else if (blockIdx.x == 0) {
			refAbove[threadIdx.x + block_size] = currentFrame->[col + width * (row - 1)];
			refAbove[threadIdx.x + block_size*2] = currentFrame->cr[col + block_size + width * (row - 1)];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->cr[col + width * (row - 1)];
			}
		}
		else if (blockIdx.y == gridDim.y - 1 && blockIdx.x == gridDim.x - 1) {
			if (width % block_size == 0 && height % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->cr[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->cr[width - 1 + width * row];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (height % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->cr[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->cr[col + block_size + width * (row - 1)];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.x == gridDim.x - 1 ) {
			if (width % block_size == 0) {
				refAbove[threadIdx.x + block_size] = currentFrame->cr[col + width * (row - 1)];
				refAbove[threadIdx.x + block_size*2] = currentFrame->cr[width - 1 + width * (row - 1)];
				if (threadIdx.x == 0) {
					refAbove[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
				}
			}
		}
		else {
			refAbove[threadIdx.x + block_size] = currentFrame->cr[col + width * (row - 1)];
			refAbove[threadIdx.x + block_size*2] = currentFrame->cr[col + block_size + width * (row - 1)];
			if (threadIdx.x == 0) {
				refAbove[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
			}
		}
	}

	// Initialize left refence array
	if (threadIdx.x == 0) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			refLeft[threadIdx.y + block_size] = 128;
			refLeft[threadidx.y + block_size*2] = 128;
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = 128;
			}
		}
		else if (blockIdx.y == 0) {
			refLeft[threadIdx.y + block_size] = currentFrame->cr[col - 1 + width * row];
			refLeft[threadidx.y + block_size*2] = currentFrame->cr[col - 1 + width * (row + block_size)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->cr[col - 1 + width * row];
			}
		}
		else if (blockIdx.x == 0) {
			refLeft[threadIdx.y + block_size] = currentFrame->cr[col + width * (blockIdx.y * blockDim.y - 1)];
			refLeft[threadidx.y + block_size*2] = currentFrame->cr[col + width * (blockIdx.y * blockDim.y - 1)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->cr[col + width * (blockIdx.y * blockDim.y - 1)];
			}
		}
		else if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
			if (width % block_size == 0 && height % block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->cr[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->cr[col + width * (height - block_size - 1)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (height % block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->cr[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->cr[col - 1 + width * (height - 1)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
				}
			}
		}
		else if (blockIdx.x == gridDim.x - 1) {
			if (width & block_size == 0) {
				refLeft[threadIdx.y + block_size] = currentFrame->cr[col - 1 + width * row];
				refLeft[threadIdx.y + block_size*2] = currentFrame->cr[col - 1 + width * (row + block_size)];
				if (threadIdx.y == 0) {
					refLeft[-1 + block_size] = currentFrame->cr[col - 1 + width * (row - 1)];
				}
			}
		}
		else {
			refLeft[threadIdx.y + block_size] = currentFrame->cr[col - 1 + width * row];
			refLeft[threadIdx.y + block_size*2] = currentFrame->cr[col - 1 + width * (row + block_size)];
			if (threadIdx.y == 0) {
				refLeft[-1 + block_size] = currentFrame->cr[col - 1 + width * (row -1)];
			}
		}
	}

	__syncthreads();

	// Filter if required
	if (filterFlag) {
		if (threadIdx.y == 0) {
			refAbove[threadIdx.x + block_size] = (refAbove[threadIdx.x - 1 + block_size] + refAbove[threadIdx.x + block_size] * 2 + refAbove[threadIdx.x + 1 + block_size]) >> 2;
			if (threadIdx.x != block_size - 1) {
				refAbove[threadIdx.x * 2 + block_size] = (refAbove[threadIdx.x * 2 - 1 + block_size] + refAbove[threadIdx.x * 2 + block_size] * 2 + refAbove[threadIdx * 2 + 1 + block_size]) >> 2;
			}
		}
		if (threadIdx.x == 0) {
			refAbove[threadIdx.y + block_size] = (refAbove[threadIdx.y - 1 + block_size] + refAbove[threadIdx.y + block_size] * 2 + refAbove[threadIdx.y + 1 + block_size]) >> 2;
			if (threadIdx.y != block_size - 1) {
				refAbove[threadIdx.y * 2 + block_size] = (refAbove[threadIdx.y * 2 - 1 + block_size] + refAbove[threadIdx.y * 2 + block_size] * 2 + refAbove[threadIdx.y * 2 + 1 + block_size]) >> 2;
			}
		}
	}

	__syncthreads();

	// Set Main and side reference arrays
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

	// Populate the prediction frames with data from the reference arrays
	if (intraPredAngle == 0) {
		predictionFrames[iMode].cr[col + row * width] = refMain[threadIdx.x + 1];
	}
	else {
		int deltaPos = (threadIdx.y + 1) * intraPredAngle;
		int deltaInt = deltaPos >> 5;
		int deltaFract = deltaPos & 31;
		
		if (deltaFract) {
			refMainIndex = threadIdx.x + deltaInt + 1;
			predictionFrames[iMode].cr[col + row * width] = ((32 - deltaFract) * refMain[refMainIndex] + deltaFract * refMain[refMainIndex + 1] + 16) >> 5;
		}
		else {
			predictionFrames[iMode].cr[col + row * width] = refMain[threadIdx.x + deltaInt + 1];
		}
	}

	// Flip if horizontal mode
	if (modeHor) {
		uint8_t tmp;
		tmp = predictionFrame[iMode].cr[col + row * width];
		__syncthreads();
		predictionFrame[iMode].cr[(blockIdx.x * blockDim.x + threadIdx.y) + (blockIdx.y * blockDim.y + threadIdx.x) * width] = tmp;
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
			dim3 predictionDimBlock(n, n, 2); // 2 for Planar and DC, angular done separately
			
			
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
			
			// Angular is done separately
			for (int i = 2; i < 35; i++) {
				AngularPredictionInitialize(currentFrame_h, predictionFrames, i, n);
			}
			
			// Call kernel to perform SATD on each block within the frame
			SATD(currentFrame, predictionFrames, &results[j]);
			// Copy result into the results array
		}
	}

	// Free allocated memory

	return results;
}
