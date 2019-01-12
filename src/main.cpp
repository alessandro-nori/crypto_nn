#include <iostream>
#include "../../FHE.h"
#include <NTL/lzz_pXFactoring.h>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "../include/layer.h"
#include "../include/relu.h"

#define MAX 55

int quantize(float x, int maxq = 127, float max = 1) {
  return int(x*maxq/max);
}

int main(int argc, char **argv) {

    if (argc < 2) {
      cerr << "usage: " << argv[0] << " <weights_file_path>" << endl;
      exit(-1);
    }

    long m=0, p=1308355967371729  , r=1; // Native plaintext space
    // Computations will be 'modulo p'
    long L=11;          // Levels
    long c=3;           // Columns in key switching matrix
    long w=64;          // Hamming weight of secret key
    long d=0;
    long security = 80;
    ZZX G;
    m = FindM(security,L,c,p, d, 0, 0);

    FHEcontext context(m, p, r);
    // initialize context
    buildModChain(context, L, c);
    // modify the context, adding primes to the modulus chain
    FHESecKey secretKey(context);
    // construct a secret key structure
    const FHEPubKey& publicKey = secretKey;

    // an "upcast": FHESecKey is a subclass of FHEPubKey

    //if(0 == d)
    G = context.alMod.getFactorsOverZZ()[0];

   secretKey.GenSecKey(w);
   // actually generate a secret key with Hamming weight w

   cout << "Generated secret key" << endl;

   //
   int n_input = 2, n_H = 2, n_output = 1;

   ifstream f;
   string file_name = argv[1];
   f.open(file_name);
   if (!f) {
     cout << "can't read from file " << file_name << endl;
     return -1;
   }

   vector<vector<int>> weights1(n_H);  // weights layer1
   vector<int> bias1(n_H); // bias layer 1
   for (int i=0; i<n_H; i++) {
     float w1, w2, b;
     f >> w1;
     f >> w2;
     f >> b;
     vector<int> w(n_input);
     w[0] = quantize(w1);
     w[1] = quantize(w2);
     bias1[i] = quantize(b);
     cout << w[0] << " " << w[1] << " " << bias1[i] << endl;
     weights1[i] = w;
   }

   vector<vector<int>> weights2(n_output);
   vector<int> bias2(n_output);
   for (int i=0; i<n_output; i++) {
     float w1, w2, b;
     f >> w1;
     f >> w2;
     f >> b;
     vector<int> w(n_H);
     w[0] = quantize(w1);
     w[1] = quantize(w2);
     bias2[i] = quantize(b);
     cout << w[0] << " " << w[1] << " " << bias2[i] << endl;
     weights2[i] = w;
   }

   f.close();

   layer l1(2, 2, 0, weights1, bias1);
   layer l2(2, 1, 0, weights2, bias2, get_scale());

   long i1 = 10;
   long i2 = 13;

   Ctxt ctx1(publicKey);
   Ctxt ctx2(publicKey);

   publicKey.Encrypt(ctx1, to_ZZX(i1));
   publicKey.Encrypt(ctx2, to_ZZX(i2));

   vector<Ctxt> inputC;
   inputC.push_back(ctx1);
   inputC.push_back(ctx2);

   vector<long> input;
   input.push_back(i1);
   input.push_back(i2);

   vector<Ctxt> outputC = relu(l1.feed_forward(inputC));
   vector<long> output = relu(l1.feed_forward(input));


   outputC = l2.feed_forward(outputC);
   output = l2.feed_forward(output);

   cout << "output1 da input non cifrato: " << output[0] << endl;

   ZZX outputZ;
   secretKey.Decrypt(outputZ, outputC[0]);
   cout << "output1 da input cifrato: " << coeff(outputZ, 0) << endl;


   return 0;
}
