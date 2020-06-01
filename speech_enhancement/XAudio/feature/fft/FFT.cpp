#include "FFT.h"

void zero_fft(double *data, int size, int fftLen, complex<double>* vec) // This function does zero padding and FFT
{
	for (int i = 0; i < fftLen; i++)     // This step does zero padding
	{
		if (i < size)
		{
			vec[i] = complex<double>(data[i]);
		}
		else
		{
			vec[i] = complex<double>(0);
		}
	}
	FFT(fftLen, vec);    // Compute FFT
}


void FFT( unsigned long fftlen, complex<double>* fftbuffer)
{
	unsigned long ulPower = 0;
	unsigned long fftlen1 = fftlen - 1;
	while (fftlen1 > 0)
	{
		ulPower++;
		fftlen1 = fftlen1 / 2;
	}

	bitset<sizeof(unsigned long) * 8> bsIndex;
	unsigned long ulIndex;
	unsigned long ulK;
	for (unsigned long p = 0; p < fftlen; p++)
	{
		ulIndex = 0;
		ulK = 1;
		bsIndex = bitset<sizeof(unsigned long) * 8>(p);
		for (unsigned long j = 0; j < ulPower; j++)
		{
			ulIndex += bsIndex.test(ulPower - j - 1) ? ulK : 0;
			ulK *= 2;
		}

		if (ulIndex > p)
		{
			complex<double> c = fftbuffer[p];
			fftbuffer[p] = fftbuffer[ulIndex];
			fftbuffer[ulIndex] = c;
		}
	}

	complex<double>* vecW = new complex<double>[fftlen / 2];
	for (unsigned long i = 0; i < fftlen / 2; i++)
	{
		vecW[i] = complex<double>(cos(2 * i * PI / fftlen), -1 * sin(2 * i * PI / fftlen));
	}

	unsigned long ulGroupLength = 1;
	unsigned long ulHalfLength = 0;
	unsigned long ulGroupCount = 0;
	complex<double> cw;
	complex<double> c1;
	complex<double> c2;
	for (unsigned long b = 0; b < ulPower; b++)
	{
		ulHalfLength = ulGroupLength;
		ulGroupLength *= 2;
		for (unsigned long j = 0; j < fftlen; j += ulGroupLength)
		{
			for (unsigned long k = 0; k < ulHalfLength; k++)
			{
				cw = vecW[k * fftlen / ulGroupLength] * fftbuffer[j + k + ulHalfLength];
				c1 = fftbuffer[j + k] + cw;
				c2 = fftbuffer[j + k] - cw;
				fftbuffer[j + k] = c1;
				fftbuffer[j + k + ulHalfLength] = c2;
			}
		}
	}
	delete[] vecW;
}
