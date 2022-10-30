#pragma once
#include "Array.h"
class MLP
{
	int n_output, n_feature, n_hidden, iterator, batch;
	double lambda1, lambda2, alpha, theta, decay;
	Array W1, W2;
	Array a1, a2, a3, z2, z3;
public:
	MLP(int output, int features, int hidden_units, int iterators,
		int batch, double alpha, double theta, double decrese_count);
	~MLP();
	void free();
	void Weight_init();
	Array encode(const Array& y);
	Array sigmoid(const Array& z);
	Array sigmoid_gradient(const Array& z);
	Array softmax(const Array& z);
	Array addBias(const Array& X, int opt = 0);
	void feedforward(const Array& X);
	Array getLog(const Array& arr);
	double sum(const Array& arr);
	double getCost(Array& y_encode, const Array& output);
	void getGradient(Array& grad1, Array& grad2, Array& y_encode);
	Array predict(const Array& X);
	void Save();
	void fit(Array& X, Array& y);
	void LoadWeight();
};