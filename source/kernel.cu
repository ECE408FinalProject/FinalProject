ece408_intra_pred_result *intra_frame_prediction(ece408_frame *imgs, int num_frames) {
	ece408_intra_pred_result *results = new ece408_intra_pred_result results[num_frames];

	// Declare GPU memory to store one "real" frame  in the video
	// Declare GPU Memory for 35 temporary frames (1 for each intra prediction mode)


	// For each frame, call the kernels	
	for (int i = 0; i < num_frames; i++) {
		// Copy the current frame into the GPU memory (try to optimize with constant memory)

		// Call kernel to populate all 35 frames with the 35 addressing modes

		// Call kernel to perform SATD on each block within the frame

		// Copy result into the results array
	}

	return results;
}

/* Possible Optimizations
 * 1. Shared memory for each thread block to store each block in the frame
 * 2. Constant memory for the entire frame for reference
 */ 