#define _USE_MATH_DEFINES
#define CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include "MLP.h"
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;


void training() {
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
	MLP nn(34, 784, 200, 1000, 20, 0.005, 0.01, 0.0001);
	nn.fit(X, y);
	nn.Save();
	Array prey = nn.predict(X);
	for (int i = 0; i < prey.row; i++) {
		for (int j = 0; j < prey.col; j++) {
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
	for (int j = 0; j < y.col; j++) {
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
	for (int i = 0; i < prey.row; i++) {
		for (int j = 0; j < prey.col; j++) {
			if (prey.Data[i][j] == y.Data[0][i * prey.col + j]) sum++;
		}
	}
	cout << endl << (sum / (NumOfData * 1.0)) * 100 << "%" << endl;
	cout << endl;
	data.free();
	X.free();
	y.free();
	prey.free();
	nn.free();
	_CrtDumpMemoryLeaks();
}

class Recognition {
	IplImage* srcImg;
	IplImage* dstImg;
	IplImage* temp;
	IplImage* plate;
	unsigned char* srcBuff;
	unsigned char* dstBuff;
	unsigned char* tBuff;
	int Width;
	int Height;
	int cornerX1, cornerX2, cornerY1, cornerY2;
	double R = 100;
	void sobel() {
		//邊緣
		int OFFSET = 1;
		int MV[3][3] = {
			{  1,  2,  1 },
			{  0,  0,  0 },
			{ -1, -2, -1 }
		};
		int MH[3][3] = {
			{ 1, 0, -1 },
			{ 2, 0, -2 },
			{ 1, 0, -1 }
		};
		for (int y = OFFSET; y < Height - OFFSET; y++) {
			for (int x = OFFSET; x < Width - OFFSET; x++) {
				int sum = 0;
				for (int i = -OFFSET; i <= OFFSET; i++) {
					for (int j = -OFFSET; j <= OFFSET; j++) {
						sum += srcBuff[(y + i) * Width + x + j] * MV[i + 1][j + 1];
						sum += srcBuff[(y + i) * Width + x + j] * MH[i + 1][j + 1];
					}
				}
				if (sum > 255) sum = 255;
				if (sum < 0) sum = 0;
				dstBuff[y * Width + x] = sum;
			}
		}
	}
	void Frame() {
		bool none = true, find = false;
		for (int y = Height - 1; y >= 0; y--) {
			none = true;
			for (int x = 0; x < Width; x++) {
				int idx = y * Width + x;
				if (dstBuff[idx] == 255) {
					find = true;
					none = false;
					if (x > cornerX2) cornerX2 = x;
					if (y > cornerY2) cornerY2 = y;
					if (x < cornerX1) cornerX1 = x;
					if (y < cornerY1) cornerY1 = y;
				}
			}
			if (none && find) break;
		}
		cornerY2 += 20;
		cornerY1 -= 20;
		cornerX1 -= 5;
		plate = cvCreateImage(cvSize(cornerX2 - cornerX1, cornerY2 - cornerY1), srcImg->depth, srcImg->nChannels);
		for (int y = cornerY1; y < cornerY2; y++) {
			for (int x = cornerX1; x < cornerX2; x++) {
				plate->imageData[(y - cornerY1) * plate->widthStep + x - cornerX1] = srcBuff[y * Width + x];
			}
		}
	}

	void Binarization(unsigned char* buff, int wid, int hei, int T) {
		//二值化
		for (int y = 0; y < hei; y++) {
			for (int x = 0; x < wid; x++) {
				int idx = y * wid + x;
				if (buff[idx] > T) buff[idx] = 255;
				else buff[idx] = 0;
			}
		}
	}
	void H_Dilation(int T) {
		//水平擴張
		unsigned char* tempBuff = new unsigned char[Height * Width];
		for (int y = 0; y < Height; y++) {
			int start = -1;
			for (int x = 0; x < Width - T - 1; x++) {
				int idx = y * Width + x;
				if (dstBuff[idx] == 255 && start == -1) {
					start = x;
				}
				if (start != -1 && x - start > T)
					start = -1;
				else if (dstBuff[idx] == 255) {
					for (int i = start; i <= x; i++) {
						tempBuff[y * Width + i] = 255;
					}
					start = x;
				}
			}
			for (int x = 0; x < Width - T - 1; x++) {
				int idx = y * Width + x;
				if (tempBuff[idx] == 255)
					dstBuff[idx] = tempBuff[idx];
				else dstBuff[idx] = 0;
			}
		}
		delete[] tempBuff;
		tempBuff = NULL;
	}
	void H_Erosion(int T) {
		//水平侵蝕
		for (int y = 0; y < Height; y++) {
			int start = -1;
			for (int x = 0; x < Width; x++) {
				int idx = y * Width + x;
				if (dstBuff[idx] == 255 && start == -1) {
					start = x;
				}
				else if ((x == Width - 1 || (dstBuff[idx] != 255 && x - start < T)) && start != -1) {
					for (int i = start; i <= x; i++) {
						dstBuff[y * Width + i] = 0;
					}
					start = -1;
				}
				if (dstBuff[idx] != 255 && x - start >= T && start != -1) start = -1;
			}
		}
	}
	void V_Dilation(int T) {
		//垂直擴張
		for (int x = 0; x < Width; x++) {
			int start = -1;
			for (int y = 0; y < Height - T - 1; y++) {
				int idx = y * Width + x;
				if (dstBuff[idx] == 255 && start == -1) start = y;
				if (start != -1 && y - start > T)
					start = -1;
				else if (dstBuff[idx] == 255 && y - start < T) {
					for (int i = start; i <= y; i++) {
						dstBuff[i * Width + x] = 255;
					}
					start = y;
				}
				if (dstBuff[idx] != 255 && y - start >= T && start != -1) start = -1;
			}
		}
	}
	void V_Erosion(int T) {
		//垂直侵蝕
		for (int x = 0; x < Width; x++) {
			int start = -1;
			for (int y = 0; y < Height; y++) {
				int idx = y * Width + x;
				if (dstBuff[idx] == 255 && start == -1) start = y;
				else if ((y == Height - 1 || (dstBuff[idx] != 255 && y - start < T)) && start != -1) {
					for (int i = start; i <= y; i++) {
						dstBuff[i * Width + x] = 0;
					}
					start = -1;
				}
				if (dstBuff[idx] != 255 && y - start >= T && start != -1) start = -1;
			}
		}
	}
	void Median() {
		unsigned char v[9];
		unsigned char* tempBuff = new unsigned char[Width * Height];
		for (int y = 1; y < Height - 1; y++) {
			for (int x = 1; x < Width - 1; x++) {
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						v[(i + 1) * 3 + j + 1] = dstBuff[(y + i) * Width + x + j];
					}
				}
				Bsort(v, 9);
				tempBuff[y * Width + x] = v[4];
			}
		}
		for (int y = 1; y < Height - 1; y++) {
			for (int x = 1; x < Width - 1; x++) {
				dstBuff[y * Width + x] = tempBuff[y * Width + x];
			}
		}
	}
	void Bsort(unsigned char v[], int size) {
		for (int i = size - 1; i > 0; i--) {
			for (int j = 0; j < i; j++) {
				if (v[j] > v[j + 1]) {
					int tempBuff = v[j];
					v[j] = v[j + 1];
					v[j + 1] = tempBuff;
				}
			}
		}
	}
public:
	Array P;
	Recognition() :srcImg(NULL), dstImg(NULL), srcBuff(NULL),
		dstBuff(NULL), temp(NULL), tBuff(NULL), plate(NULL), Width(-1), Height(-1),
		cornerX1(1000), cornerX2(-1), cornerY1(1000), cornerY2(-1) {
		P.ones(7, 784);
	}
	~Recognition() {
		cvReleaseImage(&srcImg);
		cvReleaseImage(&dstImg);
		cvReleaseImage(&temp);
		cvReleaseImage(&plate);
		srcImg = NULL;
		dstImg = NULL;
		srcBuff = NULL;
		dstBuff = NULL;
		tBuff = NULL;
	}
	void LoadImg(const char file[]) {
		temp = cvLoadImage(file, 0);
		srcImg = cvCreateImage(cvSize(640, 480), temp->depth, temp->nChannels);
		cvResize(temp, srcImg, CV_INTER_CUBIC);
		dstImg = cvCreateImage(cvSize(640, 480), temp->depth, temp->nChannels);
		srcBuff = (unsigned char*)srcImg->imageData;
		dstBuff = (unsigned char*)dstImg->imageData;
		tBuff = (unsigned char*)temp->imageData;
		Height = 480;
		Width = 640;
		cvShowImage("srcImg", srcImg);
		cvWaitKey();
	}
	void test() {
		unsigned char* buff = (unsigned char*)plate->imageData;
		int wid = plate->widthStep;
		int hei = plate->height;
		Binarization(buff, wid, hei, 100);
		cvShowImage("車牌", plate);
		cvWaitKey();
		int up = 1000, down = -1, left = 1000, right = -1;
		for (int y = 0; y < hei; y++) {
			int c = 0;
			for (int x = 0; x < wid; x++) {
				int idx = y * wid + x;
				if (abs(buff[idx] - buff[idx + 1]) == 255) {
					c++;
					if (c > 15) {

						if (y < up) up = y;
						if (y > down) down = y;
					}
				}
			}
		}
		up -= 3;
		down += 3;
		cvReleaseImage(&temp);
		temp = cvCreateImage(cvSize(wid, down - up), plate->depth, plate->nChannels);
		unsigned char* Cbuff = (unsigned char*)temp->imageData;
		for (int y = up; y < down; y++) {
			for (int x = 0; x < wid + 1; x++) {
				Cbuff[(y - up) * wid + x] = buff[y * wid + x];
			}
		}
		cvShowImage("車牌", temp);
		cvWaitKey();
		Cbuff = NULL;
	}
	void Cutting() {
		unsigned char* buff = (unsigned char*)temp->imageData;
		int wid = temp->widthStep;
		int hei = temp->height;

		IplImage* C = cvCreateImage(cvSize(28 * 7, 28), plate->depth, plate->nChannels);
		cvResize(temp, C);
		buff = (unsigned char*)C->imageData;
		cvShowImage("車牌", C);
		cvWaitKey();
		/*------------------------------------
		int *sumBlack = new int[C->widthStep];
		for (int i = 0; i < C->widthStep; i++)
			sumBlack[i] = 0;

		for (int x = 0; x < C->widthStep; x++)
			for (int y = 0; y < C->height; y++) {
				long idx = y * C->widthStep + x;
				if (C->imageData[idx] == 0) {
					sumBlack[x]++;
				}
			}
		for (int i = 0; i < C->widthStep; i++)
			cout << sumBlack[i] << " ";
		IplImage *Cutest;
		for (int i = 0, about = 0; i < C->widthStep; i++) {
			int width[2];
			if (sumBlack[i] > 0 && about == 0) {
				width[about] = i;
				about++;
			}
			else if (sumBlack[i] == 0 && about == 1) {
				width[about] = i;
				about++;
			}
			if (about == 2) {
				Cutest = cvCreateImage(CvSize(width[1] - width[0], C->height), C->depth, C->nChannels);
				if (width[1] - width[0] < 5) {
					int tmpS = 0;
					for (int i = width[0]; i < width[1]; i++)
						tmpS += sumBlack[i];
					if (tmpS < 80) {
						about = 0;
						cvReleaseImage(&Cutest);
						continue;
					}
				}
				for (int y = 0; y < Cutest->height; y++)
					for (int x = 0; x < Cutest->widthStep; x++) {
						long idx = y * Cutest->widthStep + x;
						Cutest->imageData[idx] = C->imageData[y * C->widthStep + x + width[0]];
					}
				cvShowImage("ww", Cutest);
				cvWaitKey(0);
				about = 0;
				cvReleaseImage(&Cutest);
			}
		}
		------------------------------------*/

		for (int i = 0; i < 7; i++) {
			for (int y = 0; y < 28; y++) {
				for (int x = 28 * i; x < 28 * (i + 1); x++) {
					P.Data[i][y * 28 + x - 28 * i] = (int)buff[y * 28 * 7 + x];
				}
			}
		}
		cvReleaseImage(&C);
	}

	void showImg() {
		sobel();
		cvShowImage("邊緣", dstImg);
		cvWaitKey();
		Binarization(dstBuff, Width, Height, 200);
		cvShowImage("二值化", dstImg);
		cvWaitKey();
		Median();
		cvShowImage("濾波", dstImg);
		cvWaitKey();
		V_Dilation(15);
		cvShowImage("垂直擴張", dstImg);
		cvWaitKey();
		H_Dilation(30);
		cvShowImage("水平擴張", dstImg);
		cvWaitKey();
		H_Erosion(50);
		cvShowImage("水平侵蝕", dstImg);
		cvWaitKey();
		V_Dilation(8);
		cvShowImage("垂直擴張", dstImg);
		cvWaitKey();
		H_Erosion(60);
		cvShowImage("水平侵蝕", dstImg);
		cvWaitKey();
		V_Erosion(25);
		cvShowImage("垂直侵蝕", dstImg);
		cvWaitKey();
		H_Erosion(60);
		cvShowImage("垂直侵蝕", dstImg);
		cvWaitKey();
		Frame();
		cvShowImage("車牌位置", plate);
		cvWaitKey();
		Cutting();
		test();
		Cutting();
	}
};

void Identification() {
	MLP nn(34, 784, 200, 1000, 20, 0.005, 0.01, 0.0001);
	nn.LoadWeight();
	Recognition A;
	A.LoadImg("D:/School/Digital_image_processing/Identification/x64/Debug/11.jpg");
	A.showImg();

	Array prey = nn.predict(A.P);
	for (int i = 0; i < prey.row; i++) {
		for (int j = 0; j < prey.col; j++) {
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
}

int main() {
	training();
	Identification();
	system("pause");
	return 0;
}