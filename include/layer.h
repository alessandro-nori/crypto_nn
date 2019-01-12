#ifndef _layer_H
#define _layer_H

#include "../HElib/FHE.h"

class layer {
private:
  uint8_t n_input, n_output;

  // weight matrix
  vector<vector<int>> w;
  // bias array
  vector<int> b;

  int64_t scale;

  layer();

public:
  layer(uint8_t n_input, uint8_t n_output, uint8_t af, vector<vector<int>>& w, vector<int>& b, int64_t scale = 1);
  vector<Ctxt> feed_forward(vector<Ctxt> input);
  vector<long> feed_forward(vector<long> input);
};

#endif
