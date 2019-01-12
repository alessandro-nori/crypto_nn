#include "../include/layer.h"
#include <NTL/lzz_pXFactoring.h>

using namespace std;

layer::layer(uint8_t n_input, uint8_t n_output, uint8_t af, vector<vector<int>>& w, vector<int>& b, int64_t scale) : n_input(n_input), n_output(n_output), af(af), w(w), b(b), scale(scale) {}

vector<Ctxt> layer::feed_forward(vector<Ctxt> input) {
  vector<Ctxt> output;

  for (uint8_t i=0; i<n_output; i++) {
    for (uint8_t j=0; j<n_input; j++) {
      Ctxt temp = input[j];
      temp.multByConstant(to_ZZX(w[i][j]));
      if (j==0) output.push_back(temp);
      else output[i] += temp;
    }

    output[i].addConstant(to_ZZX((long)b[i]*scale));

    // if (act_func) {
    //   Ctxt o = output[i];
    //   o.multByConstant(to_ZZX(c1));
    //
    //   Ctxt poly = o;
    //
    //   Ctxt a = output[i];
    //   // a.multiplyBy(a);
    //   a*=a;
    //   a.multByConstant(to_ZZX(c2));
    //
    //   poly.addConstant(to_ZZX(c0));
    //   poly += a;
    //   output[i] = poly;
    // }
  }

  return output;
}

vector<long> layer::feed_forward(vector<long> input) {
  vector<long> output(n_output, 0);

  for (uint8_t i=0; i<n_output; i++) {
    for (uint8_t j=0; j<n_input; j++) {
      output[i] += input[j]*w[i][j];
      // cout << output[i] << " " << input[j] << " " << int(w[i][j]) << " " << int(b[i]) << endl;
    }

    output[i] += (long)b[i]*scale;

    // if (act_func) {
    //   long temp = output[i];
    //   temp = temp*temp*c2 + temp*c1 + c0;
    //   // temp = temp*c1 + c0;
    //   output[i] = temp;
    // }
  }

  return output;
}
