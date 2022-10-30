#define _USE_MATH_DEFINES
#define CRTDBG_MAP_ALLOC
#include <iostream>
#include <fstream>
#include <sstream>
#include "MLP.h"
#include "Array.h"

using namespace std;


MLP::MLP(int output, int features, int hidden_units, int iterators,
	int batch, double alpha, double theta, double decrese_count) :
	n_output(output), n_feature(features), n_hidden(hidden_units),
	iterator(iterators), batch(batch), theta(theta), alpha(alpha),
	decay(decrese_count), W1(n_hidden, n_feature + 1),
	W2(n_output, n_hidden + 1)
{
	Weight_init();
}
MLP::~MLP() {
	free();
}
void MLP::free() {
	W1.free();
	W2.free();
	a1.free();
	a2.free();
	a3.free();
	z2.free();
	z3.free();
}
void MLP::Weight_init() {
	srand((unsigned int)time(NULL));
	//initialize W
	for (int i = 0; i < n_hidden; i++) {
		for (int j = 0; j < n_feature + 1; j++) {
			W1.Data[i][j] = (double)rand() * 2 / RAND_MAX - 1;
		}
	}
	for (int i = 0; i < n_output; i++) {
		for (int j = 0; j < n_hidden + 1; j++) {
			W2.Data[i][j] = (double)rand() * 2 / RAND_MAX - 1;
		}
	}
}
Array MLP::encode(const Array& y) {
	Array temp(n_output, y.col);
	for (int i = 0; i < y.col; i++) {
		temp.Data[(int)y.Data[0][i]][i] = 1;
	}
	return temp;
}
Array MLP::sigmoid(const Array& z) {
	Array temp(z.row, z.col);
	for (int i = 0; i < z.row; i++) {
		for (int j = 0; j < z.col; j++) {
			temp.Data[i][j] = 1.0 / (1 + pow(M_E, -z.Data[i][j]));
		}
	}
	return temp;
}
Array MLP::sigmoid_gradient(const Array& z) {
	Array sg = sigmoid(z);
	for (int i = 0; i < z.row; i++) {
		for (int j = 0; j < z.col; j++) {
			sg.Data[i][j] *= (1 - sg.Data[i][j]);
		}
	}
	return sg;
}
Array MLP::softmax(const Array& z) {
	Array temp(z.row, z.col);
	Array total(1, z.col);
	for (int i = 0; i < z.row; i++) {
		for (int j = 0; j < z.col; j++) {
			total.Data[0][j] += pow(M_E, z.Data[i][j]);
		}
	}
	for (int i = 0; i < z.row; i++) {
		for (int j = 0; j < z.col; j++) {
			temp.Data[i][j] = pow(M_E, z.Data[i][j]) / total.Data[0][j];
		}
	}
	return temp;
}
//Array softmax_gradient(const Array& z) {
//	Array sg = sigmoid(z);
//	for (int i = 0; i < z.col; i++) {
//		for (int j = 0; j < z.row; j++) {
//			sg.Data[i][j] *= (1 - sg.Data[i][j]);
//		}
//	}
//	return sg;
//}

Array MLP::addBias(const Array& X, int opt) {
	//opt 0:col 1:row
	if (opt == 0) {
		Array newX(X.row, X.col + 1);
		for (int i = 0; i < X.row; i++) {
			newX.Data[i][0] = 1;
			for (int j = 1; j < X.col + 1; j++) {
				newX.Data[i][j] = X.Data[i][j - 1];
			}
		}
		return newX;
	}
	Array newX(X.row + 1, X.col);
	for (int i = 0; i < X.row + 1; i++) {
		for (int j = 0; j < X.col; j++) {
			if (i == 0)
				newX.Data[i][j] = 1;
			else newX.Data[i][j] = X.Data[i - 1][j];
		}
	}
	return newX;
}
void MLP::feedforward(const Array& X) {
	a1 = addBias(X, 0);
	z2 = W1.dot(a1, true);
	a2 = sigmoid(z2);
	a2 = addBias(a2, 1);
	z3 = W2.dot(a2);
	a3 = sigmoid(z3);
}
Array MLP::getLog(const Array& arr) {
	Array logArr(arr.row, arr.col);
	for (int i = 0; i < arr.row; i++) {
		for (int j = 0; j < arr.col; j++) {
			logArr.Data[i][j] = log(arr.Data[i][j]);
		}
	}
	return logArr;
}
double MLP::sum(const Array& arr) {
	double sum = 0;
	for (int i = 0; i < arr.row; i++) {
		for (int j = 0; j < arr.col; j++) {
			sum += arr.Data[i][j];
		}
	}
	return sum;
}
double MLP::getCost(Array& y_encode, const Array& output) {

	double error = 0;
	for (int i = 0; i < y_encode.row; i++) {
		for (int j = 0; j < y_encode.col; j++) {
			error += (y_encode.Data[i][j] - output.Data[i][j]) * (y_encode.Data[i][j] - output.Data[i][j]);
		}
	}

	return error / output.row;
}
void MLP::getGradient(Array& grad1, Array& grad2, Array& y_encode)
{
	Array error = a3 - y_encode;
	Array delta = error * sigmoid_gradient(z3); //(34, n)
	grad2 = delta.dot(a2, true);			//(34, 201)		

	// W2.T.dot(delta) (201, 34) * (34, n)
	Array temp = W2.transpose().dot(delta);	//(201, n)
	Array g_z2 = sigmoid_gradient(z2);

	grad1.init(temp.row - 1, temp.col);	//(200, n)
	for (int i = 0; i < grad1.row; i++) {
		for (int j = 0; j < grad1.col; j++) {
			grad1.Data[i][j] = temp.Data[i + 1][j] + g_z2.Data[i][j];
		}
	}
	grad1 = grad1.dot(a1); //(200, 785)

}
Array MLP::predict(const Array& X) {
	feedforward(X);
	Array pred_y(1, a3.col);
  	for (int i = 1; i < a3.row; i++) {
		for (int j = 0; j < a3.col; j++) {
			int t = (int)pred_y.Data[0][j];
			pred_y.Data[0][j] = a3.Data[t][j] < a3.Data[i][j] ? i : t;
		}
	}
	return pred_y;
}
void MLP::Save() {
	fstream f;
	f.open("D:/School/Digital_image_processing/MLP/MLP/x64/Release/Weight.csv");
	if (!f) {
		cout << "Weight file is not found" << endl;
		return;
	}
	for (int i = 0; i < n_hidden; i++) {
		for (int j = 0; j < n_feature + 1; j++) {
			f << W1.Data[i][j] << ",";
		}
		f << "\n";
	}
	for (int i = 0; i < n_output; i++) {
		for (int j = 0; j < n_hidden + 1; j++) {
			f << W2.Data[i][j] << ",";
		}
		f << "\n";
	}
	f.close();
}
void MLP::fit(Array& X, Array& y) {
	int batch_size = X.row / batch;
	Array y_encode = encode(y);
	Array delta_w1_prev(W1.row, W1.col);
	Array delta_w2_prev(W2.row, W2.col);
	Array tempX(batch_size, X.col);
	Array tempY_en(n_output, batch_size);
	Array grad1, grad2;
	Array delta_w1;
	Array delta_w2;
	for (int i = 0; i < iterator; i++) {
		system("CLS");
		// adaptive learning rate
		double lr = theta / (1 + decay * i);
		cout << i + 1 << " / " << iterator << endl;

		for (int j = 0; j < batch; j++) {
			//batch
			for (int c = 0; c < batch_size; c++) {
				for (int r = 0; r < X.row; r++) {
					tempX.Data[c][r] = X.Data[j * batch_size + c][r];
				}
			}
			for (int o = 0; o < n_output; o++) {
				for (int r = 0; r < batch_size; r++) {
					tempY_en.Data[o][r] = y_encode.Data[o][j * batch_size + r];
				}
			}
			feedforward(tempX);
			cout << "cost: " << getCost(tempY_en, a3) << endl;
			getGradient(grad1, grad2, tempY_en);
			delta_w1 = grad1 * lr;
			delta_w2 = grad2 * lr;
			W1 -= delta_w1 + delta_w1_prev * alpha;
			W2 -= delta_w2 + delta_w2_prev * alpha;
			delta_w1_prev = delta_w1;
			delta_w2_prev = delta_w2;
		}
	}
}
void MLP::LoadWeight() {
	fstream f;
	string line;
	f.open("D:/School/Digital_image_processing/MLP/MLP/x64/Release/Weight.csv");
	if (!f) {
		cout << "開檔失敗" << endl;
		return;
	}
	for (int i = 0; i < n_hidden; i++) {
		getline(f, line, '\n');
		int n = 0, start = -1;
		for (int j = 0; j < n_feature + 1; j++) {
			string temp;
			n = line.find(",", n + 1);
			temp = temp.assign(line, start + 1, n - start);
			W1.Data[i][j] = atof(temp.c_str());
			start = n;
		}
	}
	for (int i = 0; i < n_output; i++) {
		getline(f, line, '\n');
		int n = 0, start = -1;
		for (int j = 0; j < n_hidden + 1; j++) {
			string temp;
			n = line.find(",", n + 1);
			temp = temp.assign(line, start + 1, n - start);
			W2.Data[i][j] = atof(temp.c_str());
			start = n;
		}
	}
	f.close();
}