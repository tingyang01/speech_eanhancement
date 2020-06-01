#include "windowing.h"

#include <math.h>

void hamming(double* src, int size, double* dst)
{
	double two_pi = 8.0 * atan(1.0);   // This is just 2*pi;
	double temp;
	int i;
	for (i = 0; i < size; i++)
	{
		temp = 0.54 - 0.46*cos((double)i * two_pi / (double)(size - 1.0));
		dst[i] = src[i] * temp;
	}
}

void hanning(double* src, int size, double* dst)
{
	double two_pi = 8.0 * atan(1.0);
	double temp;
	int i;
	for (i = 0; i < size; i++)
	{
		temp = 0.5 * (1.0 - cos((double)two_pi * i / (double(size) - 1.0)));
		dst[i] = src[i] * temp;
	}
}
