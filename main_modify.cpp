#define _USE_MATH_DEFINES
#define CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include "Array.h"

using namespace std;

class MLP { 
	int n_output, n_feature, n_hidden, iterator, batch;
	double lambda1, lambda2, alpha, theta, decay;
	Array W1, W2;
	Array a1, a2, a3, z2, z3;
public:
	MLP(int output, int features, int hidden_units, int iterators,
		int batch, double alpha, double theta, double decrese_count) :
		n_output(output), n_feature(features), n_hidden(hidden_units),
		iterator(iterators), batch(batch), theta(theta), alpha(alpha),
		decay(decay), W1(n_hidden, n_feature + 1),
		W2(n_output, n_hidden + 1)
	{
		Weight_init();
	}
	~MLP() {
		free();
	}
	void free() {
		W1.free();
		W2.free();
		a1.free();
		a2.free();
		a3.free();
		z2.free();
		z3.free();
	}
	void Weight_init() {
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
	Array encode(const Array& y) {
		Array temp(n_output, y.row);
		for (int i = 0; i < y.row; i++) {
			temp.Data[(int)y.Data[0][i]][i] = 1;
		}
		return temp;
	}
	Array sigmoid(const Array& z) {
		Array temp(z.col, z.row);
		for (int i = 0; i < z.col; i++) {
			for (int j = 0; j < z.row; j++) {
				temp.Data[i][j] = 1.0 / (1 + pow(M_E, -z.Data[i][j]));
			}
		}
		return temp;
	}
	Array sigmoid_gradient(const Array& z) {
		Array sg = sigmoid(z);
		for (int i = 0; i < z.col; i++) {
			for (int j = 0; j < z.row; j++) {
				sg.Data[i][j] *= (1 - sg.Data[i][j]);
			}
		}
		return sg;
	}
	Array softmax(const Array& z) {
		Array temp(z.col, z.row);
		Array total(1, z.row);
		for (int i = 0; i < z.col; i++) {
			for (int j = 0; j < z.row; j++) {
				total.Data[0][j] += pow(M_E, z.Data[i][j]);
			}
		}
		for (int i = 0; i < z.col; i++) {
			for (int j = 0; j < z.row; j++) {
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

	Array addBias(const Array& X, int opt = 0) {
		//opt 0:加行 1:加列
		if (opt == 0) {
			Array newX(X.col, X.row + 1);
			for (int i = 0; i < X.col; i++) {
				newX.Data[i][0] = 1;
				for (int j = 1; j < X.row + 1; j++) {
					newX.Data[i][j] = X.Data[i][j - 1];
				}
			}
			return newX;
		}
		Array newX(X.col + 1, X.row);
		for (int i = 0; i < X.col + 1; i++) {
			for (int j = 0; j < X.row; j++) {
				if (i == 0)
					newX.Data[i][j] = 1;
				else newX.Data[i][j] = X.Data[i - 1][j];
			}
		}
		return newX;
	}
	void feedforward(const Array& X) {
		a1 = addBias(X, 0);
		z2 = W1.dot(a1, true);
		a2 = sigmoid(z2);
		a2 = addBias(a2, 1);
		z3 = W2.dot(a2);
		a3 = sigmoid(z3);
	}
	Array getLog(const Array& arr) {
		Array logArr(arr.col, arr.row);
		for (int i = 0; i < arr.col; i++) {
			for (int j = 0; j < arr.row; j++) {
				logArr.Data[i][j] = log(arr.Data[i][j]);
			}
		}
		return logArr;
	}
	double sum(const Array& arr) {
		double sum = 0;
		for (int i = 0; i < arr.col; i++) {
			for (int j = 0; j < arr.row; j++) {
				sum += arr.Data[i][j];
			}
		}
		return sum;
	}
	double getCost(Array& y_encode, const Array& output) {		
		
		double error = 0;
		for (int i = 0; i < y_encode.col; i++) {
			for (int j = 0; j < y_encode.row; j++) {
				error += (y_encode.Data[i][j] - output.Data[i][j]) * (y_encode.Data[i][j] - output.Data[i][j]);
			}
		}

		return error / output.row;
	}
	void getGradient(Array& grad1, Array& grad2, Array& y_encode)
	{
		Array error = a3 - y_encode;
		Array delta = error * sigmoid_gradient(z3); //(34, n)
		grad2 = delta.dot(a2, true);			//(34, 201)		
		
		// W2.T.dot(delta) (201, 34) * (34, n)
		Array temp = W2.transpose().dot(delta);	//(201, n)
		Array g_z2 = sigmoid_gradient(z2);

		grad1.init(temp.col - 1, temp.row);	//(200, n)
		for (int i = 0; i < grad1.col; i++) {
			for (int j = 0; j < grad1.row; j++) {
				grad1.Data[i][j] = temp.Data[i + 1][j] + g_z2.Data[i][j];
			}
		}
		grad1 = grad1.dot(a1); //(200, 785)

	}
	Array predict(const Array& X) {
		feedforward(X);
		Array pred_y(1, a3.row);
		for (int i = 1; i < a3.col; i++) {
			for (int j = 0; j < a3.row; j++) {
				int t = (int)pred_y.Data[0][j];
				pred_y.Data[0][j] = a3.Data[t][j] < a3.Data[i][j] ? i : t;
			}
		}
		return pred_y;
	}
	void Save() {
		fstream f;
		f.open("D:/School/Digital_image_processing/MLP/MLP/x64/Release/Weight.csv");
		if (!f) {
			cout << "開檔失敗" << endl;
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
	void fit(Array& X, Array& y) {
		int batch_size = X.col / batch;
		Array y_encode = encode(y);
		Array delta_w1_prev(W1.col, W1.row);
		Array delta_w2_prev(W2.col, W2.row);
		Array tempX(batch_size, X.row);
		Array tempY_en(n_output, batch_size);
		Array grad1, grad2;
		Array delta_w1;
		Array delta_w2;
		for (int i = 0; i < iterator; i++) {
			system("CLS");
			// adaptive learning rate
			theta /= (1 + decay * i);
			cout << i + 1 << " / " << iterator << endl;

			for (int j = 0; j < batch; j++) {
				//分批訓練
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
				delta_w1 = grad1 * theta;
				delta_w2 = grad2 * theta;
				W1 -= delta_w1 + delta_w1_prev * alpha;
				W2 -= delta_w2 + delta_w2_prev * alpha;
				delta_w1_prev = delta_w1;
				delta_w2_prev = delta_w2;
			}
		}
	}
};
int main() {
	Array data;
	int NumOfData = 540;
	data.LoadData("D:/School/Digital_image_processing/MLP/Debug/ImgData.csv", NumOfData, 785);

	Array X(NumOfData, 784);
	Array y(1, NumOfData);
	for (int i = 0; i < NumOfData; i++) {
		for (int j = 0; j < 784; j++) {
			X.Data[i][j] = data.Data[i][j];			
		}
		y.Data[0][i] = data.Data[i][784];
	}
	MLP nn(34, 784, 200, 500, 20, 0.005, 0.01, 0.001);
	nn.fit(X, y);
	nn.Save();
	Array prey = nn.predict(X);
	for (int i = 0; i < prey.col; i++) {
		for (int j = 0; j < prey.row; j++) {
			if (prey.Data[i][j] >= 10) {
				cout << setw(2) << (char)(prey.Data[i][j] + 55) << " ";
			}
			else if (prey.Data[i][j] >= 18) {
				cout << setw(2) << (char)(prey.Data[i][j] + 56) << " ";
			}
			else if (prey.Data[i][j] >= 23) {
				cout << setw(2) << (char)(prey.Data[i][j] + 57) << " ";
			}
			else
				cout << setw(2) << prey.Data[i][j] << " ";
		}
		cout << endl;
	}
	for (int j = 0; j < y.row; j++) {
		if (y.Data[0][j] >= 10) {
			cout << setw(2) << (char)(y.Data[0][j] + 55) << " ";
		}
		else if (y.Data[0][j] >= 18) {
			cout << setw(2) << (char)(y.Data[0][j] + 56) << " ";
		}
		else if (y.Data[0][j] >= 23) {
			cout << setw(2) << (char)(y.Data[0][j] + 57) << " ";
		}
		else
			cout << setw(2) << y.Data[0][j] << " ";
	}
	double sum = 0;
	for (int i = 0; i < prey.col; i++) {
		for (int j = 0; j < prey.row; j++) {
			if (prey.Data[i][j] == y.Data[0][i * prey.row + j]) sum++;
		}
	}
	cout << endl << (sum / (NumOfData * 1.0)) * 100 << "%" << endl;
	cout << endl;
	data.free();
	X.free();
	y.free();
	prey.free();
	nn.free();
	system("pause");
	_CrtDumpMemoryLeaks();
	return 0;
}