//=======================================================================
/** @file MFCC.cpp
 *  @brief Calculates Mel Frequency Cepstral Coefficients
 *  @author Adam Stark
 *  @copyright Copyright (C) 2014  Adam Stark
 *
 * This file is part of the 'Gist' audio analysis library
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//=======================================================================

#include "MFCC.h"
#include <cfloat>
#include <assert.h>
#include "../smooth/savitzky_golay/SGSmooth.hpp"

//==================================================================
template <class T>
MFCC<T>::MFCC (int frameSize_, int samplingFrequency_)
{
    nMels = 40;
	numCoefficient = 13;
    frameSize = frameSize_;
    samplingFrequency = samplingFrequency_;

    initialise();
}

//==================================================================
template <class T>
void MFCC<T>::setNumCoefficients (int numCoefficients_)
{
	numCoefficient = numCoefficients_;
    initialise();
}

//==================================================================
template <class T>
void MFCC<T>::setNumMels(int nMels_)
{
	nMels = nMels_;
	initialise();
}

//==================================================================
template <class T>
void MFCC<T>::setFrameSize (int frameSize_)
{
    frameSize = frameSize_;
    initialise();
}

//==================================================================
template <class T>
void MFCC<T>::setSamplingFrequency (int samplingFrequency_)
{
    samplingFrequency = samplingFrequency_;
    initialise();
}

//==================================================================
template <class T>
void MFCC<T>::calculateMelFrequencyCepstralCoefficients (const std::vector<T>& magnitudeSpectrum, T energy)
{
    calculateMelFrequencySpectrum (magnitudeSpectrum);
	PowerToDB();  //convert melSpectrum to logSpectrum

    /*for (unsigned i = 0; i < melSpectrum.size(); i++)
        MFCCs[i] = log (melSpectrum[i] + (T)FLT_MIN);*/

    discreteCosineTransform (logSpectrum, logSpectrum.size());
	unsigned i;
	unsigned vSize = MFCCs.size();
	MFCCs[0] = log(energy);
	for (i = 1; i < vSize; i++)
		MFCCs[i] = logSpectrum[i + 1];
}

template <class T>
void MFCC<T>::PowerToDB(double ref, double amin, double top_db)
{
	/**librosa.power_to_db(...)*/
	assert(amin > 0);
	unsigned i, j; 
	for (i = 0; i < melSpectrum.size(); i++)
	{
		logSpectrum[i] = 10.0 * log10(fmax(amin, melSpectrum[i])) 
			- 10.0 * log10(fmax(amin, ref));
		if (top_db >= 0)
		{
			T maxV = -FLT_MAX;
			for (j = 0; j < logSpectrum.size(); j++)
			{
				if (maxV < logSpectrum[j]) maxV = logSpectrum[j];
			}
			logSpectrum[i] = fmax(logSpectrum[i], maxV - top_db);
		}
	}
}

//==================================================================
template <class T>
void MFCC<T>::calculateMelFrequencySpectrum (const std::vector<T>& magnitudeSpectrum)
{
    for (int i = 0; i < nMels; i++)
    {
        double coeff = 0;
        
        for (int j = 0; j < (int)magnitudeSpectrum.size(); j++)
        {
            //coeff += (T)((magnitudeSpectrum[j] * magnitudeSpectrum[j]) * filterBank[i][j]);
			coeff += (T)(magnitudeSpectrum[j] * filterBank[i][j]);
        }
        
        melSpectrum[i] = coeff;
    }
}

//==================================================================
template <class T>
void MFCC<T>::initialise()
{
    magnitudeSpectrumSize = frameSize / 2 + 1;
    minFrequency = 0;
    maxFrequency = samplingFrequency / 2;

    melSpectrum.resize (nMels);
    MFCCs.resize (numCoefficient - 1);   // 

	logSpectrum.resize(nMels);
    //calculateMelFilterBank();
	calculateMelFilterBank2();
}

//==================================================================
template <class T>
void MFCC<T>::discreteCosineTransform (std::vector<T>& inputSignal, const size_t numElements, bool bOrtho)
{
	/**
	*@bOrtho is a bool value dedicated if orthogonal norm is applied to result or not.
	*/
    // the input signal must have the number of elements specified in the numElements variable
    assert (inputSignal.size() == numElements);
    
    //T signal[numElements]; // copy to work on
	T* signal = new T[numElements];
    
    for (unsigned i = 0; i < numElements; i++)
        signal[i] = inputSignal[i];
    
    T N = (T)numElements;
    T piOverN = M_PI / N;

    for (unsigned k = 0; k < numElements; k++)
    {
        T sum = 0;
        T kVal = (T)k;

        for (unsigned n = 0; n < numElements; n++)
        {
            T tmp = piOverN * (((T)n) + 0.5) * kVal;

            sum += signal[n] * cos (tmp);
        }

        inputSignal[k] = (T)(2 * sum);
    }

	if (bOrtho)
	{
		T factor0 = sqrt(1.0 / (4.0 * numElements));
		T factor = sqrt(1.0 / (2.0 * numElements));
		unsigned i;
		inputSignal[0] *= factor0;
		for (i = 1; i < numElements; i++)
			inputSignal[i] *= factor;
	}
	delete signal;
}

//==================================================================
template <class T>
void MFCC<T>::calculateMelFilterBank()
{
    int maxMel = floor ((T)frequencyToMel (maxFrequency));
    int minMel = floor ((T)frequencyToMel (minFrequency));

    filterBank.resize (nMels);

    for (int i = 0; i < nMels; i++)
    {
        filterBank[i].resize (magnitudeSpectrumSize);

        for (int j = 0; j < magnitudeSpectrumSize; j++)
        {
            filterBank[i][j] = 0.0;
        }
    }

    std::vector<int> centreIndices;

    for (int i = 0; i < nMels + 2; i++)
    {
        double f = i * (maxMel - minMel) / (nMels + 1) + minMel;

        double tmp = log (1 + 1000.0 / 700.0) / 1000.0;
        tmp = (exp (f * tmp) - 1) / (samplingFrequency / 2);

        tmp = 0.5 + 700 * ((double)magnitudeSpectrumSize) * tmp;

        tmp = floor (tmp);

        int centreIndex = (int)tmp;

        centreIndices.push_back (centreIndex);
    }

    for (int i = 0; i < nMels; i++)
    {
        int filterBeginIndex = centreIndices[i];
        int filterCenterIndex = centreIndices[i + 1];
        int filterEndIndex = centreIndices[i + 2];

        T triangleRangeUp = (T)(filterCenterIndex - filterBeginIndex);
        T triangleRangeDown = (T)(filterEndIndex - filterCenterIndex);

        // upward slope
        for (int k = filterBeginIndex; k < filterCenterIndex; k++)
        {
            filterBank[i][k] = ((T)(k - filterBeginIndex)) / triangleRangeUp;
        }

        // downwards slope
        for (int k = filterCenterIndex; k < filterEndIndex; k++)
        {
            filterBank[i][k] = ((T)(filterEndIndex - k)) / triangleRangeDown;
        }
    }
}

//==================================================================
template <class T>
void MFCC<T>::calculateMelFilterBank2()
{
	//this is func converted from mel fun of librosa's filters.py
	
	filterBank.resize(nMels);
	for (int i = 0; i < nMels; i++)
	{
		filterBank[i].resize(magnitudeSpectrumSize);

		for (int j = 0; j < magnitudeSpectrumSize; j++)
		{
			filterBank[i][j] = 0.0;
		}
	}

	T *fftfreqs = new T[magnitudeSpectrumSize];
	int i = 0;
	T fstep = samplingFrequency / 2.0 / (magnitudeSpectrumSize - 1);
	fftfreqs[0] = 0;
	for (i = 1; i < magnitudeSpectrumSize; i++)
	{
		fftfreqs[i] = fstep * i;
	}
	
	T* mel_f = new T[nMels + 2];
	T maxMel = frequencyToMel(maxFrequency);
	T minMel = frequencyToMel(minFrequency);
	fstep = (maxMel - minMel) / (nMels + 1);
	for (i = 0; i < nMels + 2; i++)
		mel_f[i] = melToFrequency(i * fstep + minMel);

	T *fdiff = new T[nMels + 1];
	for (i = 0; i < nMels + 1; i++)
	{
		fdiff[i] = mel_f[i + 1] - mel_f[i];
	}

	//numpy.subtract.outer(mel_f, fftfreqs)//////////
	T **ramps = new T*[nMels + 2];
	for (i = 0; i < nMels + 2; i++)
	{
		ramps[i] = new T[magnitudeSpectrumSize];
	}
	int j = 0;
	for (i = 0; i < nMels + 2; i++)
	{
		for (j = 0; j < magnitudeSpectrumSize; j++)
		{
			ramps[i][j] = mel_f[i] - fftfreqs[j];
		}
	}
	/////////////////////////////////////////////////
	T lower, upper;
	for (i = 0; i < nMels; i++)
	{
		T enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
		for (j = 0; j < magnitudeSpectrumSize; j++)
		{
			lower = -ramps[i][j] / fdiff[i];
			upper = ramps[i + 2][j] / fdiff[i + 1];
			filterBank[i][j] = (T)fmax(0, fmin(lower, upper)) * enorm;
		}
	}
}

//==================================================================
template <class T>
T MFCC<T>::frequencyToMel (T frequency, bool htk)
{
	if (htk)
		return int(1127) * log (1 + (frequency / 700.0));

	// Fill in the linear part
	double f_min = 0.0;
	double f_sp = 200.0 / 3;
	double mel = (frequency - f_min) / f_sp;
	// Fill in the log-scale part
	double min_log_hz = 1000.0;							// beginning of log region (Hz)
	double min_log_mel = (min_log_hz - f_min) / f_sp;	// same (Mels)
	double logstep = log(6.4) / 27.0;					// step size for log region

	if (frequency >= min_log_hz)
		mel = min_log_mel + log(frequency / min_log_hz) / logstep;

	return (T)mel;
}

//===========================================================
template <class T>
T MFCC<T>::melToFrequency(T mel, bool htk)
{
	if (htk)
		return 700.0 * pow(10.0, (mel / 2595.0) - 1.0);

	// Fill in the linear scale
	T f_min = 0.0;
	T f_sp = 200.0 / 3;
	T freq = f_min + f_sp * mel;

	// And now the nonlinear scale
	T min_log_hz = 1000.0;						// beginning of log region(Hz)
	T min_log_mel = (min_log_hz - f_min) / f_sp;  // same(Mels)
	T logstep = log(6.4) / 27.0;		        // step size for log region

	if (mel >= min_log_mel)
	{
		freq = min_log_hz * exp(logstep * (mel - min_log_mel));
	}
	return freq;
}

//===========================================================
template class MFCC<float>;
template class MFCC<double>;
