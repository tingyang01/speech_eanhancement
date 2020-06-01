#pragma once
#include <math.h>
#include <complex> 
#include <bitset> 

const double PI = 3.1415926536;
using namespace std;

void FFT(unsigned long fftlen, complex<double>* fftbuffer);

void zero_fft(double *data, int size, int fftLen, complex<double>* fftbuffer);
