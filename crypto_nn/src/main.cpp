#include "../include/FHE.h"
#include <NTL/lzz_pXFactoring.h>

#include <fstream>
#include <iostream>

#include "../include/layer.h"
#include "../include/relu.h"

#define MAXQ 127

int quantize(float x, int maxq = MAXQ, float max = 1) {
  int xq = int(x*maxq/max);
  return xq;
}

int main(int argc, char **argv) {

    if (argc < 3) {
      cerr << "usage: " << argv[0] << " <weights_file_path> <test_dataset>" << endl;
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
     cout << "can't load neural network parameters from " << file_name << endl;
     return -1;
   }


   // initialization hidden layer
   vector<vector<int>> weights1(n_H);  // weights
   vector<int> bias1(n_H); // bias
   for (int i=0; i<n_H; i++) {
     float w1, w2, b;
     f >> w1 >> w2 >> b;
     vector<int> w(n_input);
     w[0] = quantize(w1);
     w[1] = quantize(w2);
     bias1[i] = quantize(b);
     // cout << w[0] << " " << w[1] << " " << bias1[i] << endl;
     weights1[i] = w;
   }

   // initialization output layer
   vector<vector<int>> weights2(n_output);
   vector<int> bias2(n_output);
   for (int i=0; i<n_output; i++) {
     float w1, w2, b;
     f >> w1 >> w2 >> b;
     vector<int> w(n_H);
     w[0] = quantize(w1);
     w[1] = quantize(w2);
     bias2[i] = quantize(b);
     // cout << w[0] << " " << w[1] << " " << bias2[i] << endl;
     weights2[i] = w;
   }

   f.close();

   layer l1(2, 2, weights1, bias1);
   layer l2(2, 1, weights2, bias2, get_scale()*127);

   cout << "Neural network parameters loaded" << endl;

   f.open(argv[2]);
   if (!f) {
     cout << "can't load data for test" << endl;
     return -1;
   }

   ofstream fout;
   fout.open("pred.txt");
   if (!f) {
     cout << "can't output the predictions" << endl;
     return -1;
   }

   long x1, x2, pred;
   while (f >> x1 >> x2 >> pred) {

     // this should be done client side and cipher inputs should be sent to the Cloud
     Ctxt ctx1(publicKey);
     Ctxt ctx2(publicKey);

     publicKey.Encrypt(ctx1, to_ZZX(x1));
     publicKey.Encrypt(ctx2, to_ZZX(x2));
     //

     // what the Cloud receives
     vector<Ctxt> inputC;
     inputC.push_back(ctx1);
     inputC.push_back(ctx2);

     vector<Ctxt> outputC = relu(l1.feed_forward(inputC));

     outputC = l2.feed_forward(outputC);
     // now the Cloud can send back the cipher output to the client which can decrypt it

     ZZX outputZ;
     secretKey.Decrypt(outputZ, outputC[0]);

     /*
     * plain output is scaled of 127*127 (due to weights quantization) * 10000 (due to integer coefficient of ReLU)
     * in my solution the Cloud should also send this value to the client in order to allow it to convert the output to real from integer
     */

     cout << "plain input: " << x1 << " " << x2 << endl;
     cout << "plain output: " << coeff(outputZ, 0) << endl;
     cout << endl;

     fout << x1 << " " << x2 << " " << coeff(outputZ, 0) << endl;

   }

   f.close();

   cout << "output scale: " << MAXQ*MAXQ*get_scale() << endl;

   return 0;
}
