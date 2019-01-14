#include "../include/relu.h"
#include <NTL/lzz_pXFactoring.h>

// this variable is used to get int coefficient for the ReLU approximation
static int64_t scale = 10000;

static int64_t c0 = 520000; // constant
static int64_t c1 = 5000; // coefficient for degree 1
static int64_t c2 = 12;  // coefficient for degree 2


vector<Ctxt> relu(vector<Ctxt> input) {
  vector<Ctxt> output;

  for (int i=0; i<input.size(); i++) {
    Ctxt x2 = input[i];
    x2*=x2;
    x2.multByConstant(to_ZZX(c2));

    Ctxt x1 = input[i];
    x1.multByConstant(to_ZZX(c1));

    output.push_back(x2);
    output[i] += x1;
    output[i].addConstant(to_ZZX(c0));
  }

  return output;
}

vector<int64_t> relu(vector<int64_t> input) {
  vector<int64_t> output(input.size());

  for (int i=0; i<input.size(); i++) {
    output[i] = input[i]*input[i]*c2+input[i]*c1+c0;
  }

  return output;
}

int64_t get_scale() {
  return scale;
}
