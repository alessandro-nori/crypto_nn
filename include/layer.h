#ifndef _layer_H
#define _layer_H

#include "../../FHE.h"
#include <cstdint>

class layer {
private:
  uint8_t n_input, n_output;

  // weight matrix
  vector<vector<int>> w;
  // bias array
  vector<int> b;

  // activation function
  uint8_t af;

  // int64_t c0 = 52000000000;
  // int64_t c1 = 3937008;
  // int64_t c2 = 74;
  // int c0 = 0;
  // int c1 = 1;
  // int c2 = 0;

  int64_t scale;

  bool act_func;

  layer();

public:
  layer(uint8_t n_input, uint8_t n_output, uint8_t af, vector<vector<int>>& w, vector<int>& b, int64_t scale = 1);
  vector<Ctxt> feed_forward(vector<Ctxt> input);
  vector<long> feed_forward(vector<long> input);
};

#endif
