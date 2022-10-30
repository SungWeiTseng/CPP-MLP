#pragma once
class Array
{
public:
	int col, row;
	double** Data;
	Array();
	Array(const Array& other);
	Array(int col, int row);
	~Array();
	void init(int col, int row);
	void free();
	Array operator=(const Array& other);
	Array operator+(const Array& other);
	Array operator-(const Array& other);
	Array operator*(const Array& other);
	Array operator*(double m);
	Array operator-=(const Array& other);
	Array operator+=(const Array& other);
	Array operator*=(const Array& other);
	Array operator*=(double m);
	Array dot(const Array& other, bool T = false);
	Array transpose();
	void ones(int col, int row);
	void LoadData(const char path[], int nData, int features);
};

