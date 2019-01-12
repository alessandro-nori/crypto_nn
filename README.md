# Crypto Neural Network

Crypto_nn is a very simple example of neural network that can perform classification over encrypted data using homomorphic encryption.
The idea is taken from [CryptoDL: Deep Neural Networks over Encrypted Data](https://arxiv.org/pdf/1711.05189.pdf) by Ehsan Hesamifard, Hassan Takabi, Mehdi Ghasemi where
you can find all the details.

## Activation function
To use activation functions within HE schemes, they should be approximated in a form which is implemented using only addition and multiplication (e.g. polynomial).

In this example I simulated the ReLU function as presented in the paper and I obtained the following approximation:

0.0012x<sup>2</sup> + 0.5x + 52
