#ifndef __ECE408_COMPETITION_H__
#define __ECE408_COMPETITION_H__

#include <stdint.h>
#include "x265.h"

struct ece408_frame {
  int width; //width of the frame in pixels 
  int height; //height of the frame in pixels
  uint8_t *y; //pointer to a width*height array of luma values
  uint8_t *cb; //pointer to a (width/2)*(height/2) array of cb values
  uint8_t *cr; //pointer to a (width/2)*(height/2) array of cr values
  //NOTE: y/cr/cb are all stored in row-major order

  ece408_frame() { }

  ece408_frame(int _width, int _height, x265_picture *pic)
  {
    create(_width, _height, pic);
  }

  void create(int _width, int _height, x265_picture *pic) {
    width = _width;
    height = _height;
    y = (uint8_t *)(pic->planes[0]);
    cb = (uint8_t *)(pic->planes[1]);
    cr = (uint8_t *)(pic->planes[2]);
  }
};

struct ece408_intra_pred_result {
  int luma_block_size; //Should be 8,16,32,or 64, implies chroma block size of 4, 8, 16, 32, respectively
  int num_blocks; //(frame height/luma_block_size)*(frame width/luma_block_size)
  uint8_t *y_modes; //Pointer to (35*num_blocks) array of mode indices (0-34). Each chunk of 35 indices should be the mode indices for the corresponding luma block of the frame, in ascending order of their associated SATD.
  int32_t *y_satd_results; //Pointer to (35*num_blocks) array of the actual SATD results for the corresponding luma frame block and prediction mode
  uint8_t *cb_modes; //Same as y_modes, but for chroma 1
  int32_t *cb_satd_results; //Same as y_satd_results, but for chroma 1
  uint8_t *cr_modes;//Same as y_modes, but for chroma 2
  int32_t *cr_satd_results; //Same as y_satd_results, but for chroma 2
  //NOTE: all 6 arrays are stored in row-major order of the prediction blocks

  ece408_intra_pred_result() { }

  ece408_intra_pred_result(int width, int height, int _luma_block_size)
  {
    create(width, height, _luma_block_size);
  }

  void create(int width, int height, int _luma_block_size) {
    luma_block_size = _luma_block_size;
    num_blocks = (height/luma_block_size)*(width/luma_block_size);
    y_modes = new uint8_t[35*num_blocks];
    y_satd_results = new int32_t[35*num_blocks];
    if(_luma_block_size > 4) {
    	cb_modes = new uint8_t[35*num_blocks];
    	cb_satd_results = new int32_t[35*num_blocks];
    	cr_modes = new uint8_t[35*num_blocks];
    	cr_satd_results = new int32_t[35*num_blocks];
    }
  }

  void destroy() {
    delete[] y_modes;
    delete[] y_satd_results;
    if(luma_block_size > 4) {
    	delete[] cb_modes;
    	delete[] cb_satd_results;
    	delete[] cr_modes;
    	delete[] cr_satd_results;
    }
  }
};

#endif //#ifndef __ECE408_COMPETITION_H__
