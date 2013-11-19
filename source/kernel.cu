/* Possible Optimizations
 * 1. Shared memory for each thread block to store each block in the frame
 * 2. Constant memory for the entire frame for reference
 * 3. Use streams to simultaneously compute multiple n sizes and multiple frames
 */ 

 #define NUM_MODES		35
 #define NUM_CHANNELS	3

 __device__
void DCPrediction(ece408_frame *currentFrame, ece408_frame *predictionFrames) {

}

__device__
void PlanarPrediction(ece408_frame *currentFrame, ece408_frame *predictionFrames) {

}

__device__
void AngularPrediction(ece408_frame *currentFrame, ece408_frame *predictionFrames) {

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

	// Declare GPU Memory for 35 temporary frames (1 for each intra prediction mode)
	ece408_frame *predictionFrames_d;
	cudaMalloc((void**)&predictionFrames), num_frames * frameSize);


	// For each frame, call the kernels	
	for (int i = 0; i < num_frames; i++) {
		// Note: We may need to insert another for loop here to loop different n

		ece408_frame currentFrame_h = imgs[i];

		// Set kernel dimensions
		dim3 predictionDimGrid(currentFrame_h.width, currentFrame_h.height, NUM_MODES);
		dim3 predictionDimBlock(n, n, NUM_CHANNELS);

		dim3 satdDimGrid(currentFrame_h.width, currentFrame_h.height, NUM_CHANNELS);
		dim3 satdDimBlock(n, n, NUM_MODES)

		// Copy the current frame into the GPU memory (try to optimize with constant memory)
		error = cudaMemcpy(currentFrame_d, currentFrame_h, frameSize, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) FATAL("Unable to copy memory");

		// Call kernel to populate all 35 frames with the 35 addressing modes
		intra_frame_prediction<<<predictionDimGrid, predictionDimBlock>>>(currentFrame, predictionFrames);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess) FATAL("Unable to launch/execute kernel");

		// Call kernel to perform SATD on each block within the frame

		// Copy result into the results array
	}

	// Free allocated memory

	return results;
}
