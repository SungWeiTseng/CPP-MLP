#include "Array.h"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

Array::Array() :Data(NULL), col(-1), row(-1) {}
Array::Array(const Array& other) : Data(NULL) {
	free();
	this->col = other.col;
	this->row = other.row;
	Data = new double* [row];
	for (int i = 0; i < row; i++) {
		Data[i] = new double[col];
		for (int j = 0; j < col; j++) {
			Data[i][j] = other.Data[i][j];
		}
	}
}
Array::Array(int row, int col) :Data(NULL) {
	init(row, col);
}
Array::~Array() {
	free();
}
void Array::init(int row, int col) {
	free();
	this->col = col;
	this->row = row;
	Data = new double* [row];
	for (int i = 0; i < row; i++) {
		Data[i] = new double[col] {0};
	}
}
Array Array::operator=(const Array& other) {
	init(other.row, other.col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Data[i][j] = other.Data[i][j];
		}
	}
	return *this;
}
Array Array::operator+(const Array& other) {
	Array temp(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp.Data[i][j] = Data[i][j] + other.Data[i][j];
		}
	}
	return temp;
}
Array Array::operator-(const Array& other) {
	Array temp(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp.Data[i][j] = Data[i][j] - other.Data[i][j];
		}
	}
	return temp;
}
Array Array::operator*(const Array& other) {
	Array temp(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp.Data[i][j] = Data[i][j] * other.Data[i][j];
		}
	}
	return temp;
}
Array Array::transpose() {
	Array temp(col, row);
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			temp.Data[i][j] = Data[j][i];
		}
	}
	return temp;
}

Array Array::operator*(double m) {
	Array temp(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp.Data[i][j] = Data[i][j] * m;
		}
	}
	return temp;
}
Array Array::operator-=(const Array& other) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Data[i][j] -= other.Data[i][j];
		}
	}
	return *this;
}

Array Array::operator+=(const Array& other) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Data[i][j] += other.Data[i][j];
		}
	}
	return *this;
}

Array Array::operator*=(const Array& other) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Data[i][j] *= other.Data[i][j];
		}
	}
	return *this;
}

Array Array::operator*=(double m) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Data[i][j] *= m;
		}
	}
	return *this;
}
Array Array::dot(const Array& other, bool T) {
	if (T) {
		Array temp(row, other.row);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				for (int k = 0; k < other.row; k++) {
					temp.Data[i][k] += Data[i][j] * other.Data[k][j];
				}
			}
		}
		return temp;
	}
	Array temp(row, other.col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < other.col; k++) {
				temp.Data[i][k] += Data[i][j] * other.Data[j][k];
			}
		}
	}
	return temp;
}
void Array::ones(int row, int col) {
	free();
	this->col = col;
	this->row = row;
	Data = new double* [row];
	for (int i = 0; i < row; i++) {
		Data[i] = new double[col];
		for (int j = 0; j < col; j++) {
			Data[i][j] = 1;
		}
	}
}
void Array::LoadData(const char path[], int nData, int features) {
	free();
	col = features;
	row = nData;
	fstream file;
	string line;
	file.open(path);
	if (!file) {
		cout << "file is not found" << endl;
		return;
	}
	Data = new double* [nData];
	for (int i = 0; i < nData; i++) {
		Data[i] = new double[features];
		getline(file, line, '\n');
		int n = 0, start = -1;
		for (int j = 0; j < features; j++) {
			string temp;
			n = line.find(",", n + 1);
			temp = temp.assign(line, start + 1, n - start);
			if (temp[0] >= 65 && temp[0] < 73) {
				Data[i][j] = temp[0] - 55;
			}
			else if (temp[0] >= 74 && temp[0] < 79) {
				Data[i][j] = temp[0] - 56;
			}
			else if (temp[0] >= 80) {
				Data[i][j] = temp[0] - 57;
			}
			else {
				Data[i][j] = atoi(temp.c_str());
			}
			start = n;
		}
	}
	file.close();
}
void Array::free() {
	if (Data != NULL) {
		for (int i = 0; i < row; i++) {
			delete[] Data[i];
			Data[i] = NULL;
		}
		delete Data;
		Data = NULL;
	}
}