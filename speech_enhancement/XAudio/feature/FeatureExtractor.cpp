#include "FeatureExtractor.h"
#include "smooth/savitzky_golay/SGSmooth.hpp"
#include "librosa/spectrum.h"

#define EPS 1e-15

FeatureExtractor::FeatureExtractor(int framesize, int frame_step, int samplerate)
{
	this->framesize = framesize;
	this->samplerate = samplerate;
	this->frame_step = frame_step;
	gist = new Gist<double>(frame_step*3, framesize, frame_step, samplerate);
}


FeatureExtractor::~FeatureExtractor()
{
	if (gist)
	{
		delete gist;
		gist = NULL;
	}
}


void Normalization(float* signal, int size)
{
	float fmean, fmax;
	float sum = 0.0;
	int i;
	//mean
	fmax = 0.0;
	for (i = 0; i < size; i++)
	{
		sum += signal[i];
		if (fmax < abs(signal[i]))
			fmax = abs(signal[i]);
	}
	fmean = sum / size;
	for (i = 0; i < size; i++)
	{
		signal[i] = (signal[i] - fmean) / (float)(fmax + EPS);
	}
}

float* FeatureExtractor::Padding(float* data, int size, int padding_size, PadType pad_type)
{
	float *padded = new float[size + padding_size * 2];
	switch (pad_type)
	{
	case REFLECT:
		int i;
		for (i = 0; i < padding_size; i++)
		{
			padded[i] = data[padding_size - i];
		}
		memcpy_s(padded + padding_size, sizeof(float) * size, data, sizeof(float) * size);
		for (i = 0; i < padding_size; i++)
		{
			padded[padding_size + size + i] = data[padding_size - i - 1];
		}
	default:
		break;
	}
	return padded;
}


void FeatureExtractor::stft(float* data, int size, std::vector<std::vector<std::pair<double, double>>>& stft, bool center)
{
	float *padded_data = NULL;
	int padded_size = size;
	int i = 0;
	if (center)
	{
		padded_data = Padding(data, size, framesize / 2, REFLECT);
		padded_size = size + framesize;
	}

	//double *buf = new double[padded_size];
	std::vector<double> buf(padded_size);
	for (i == 0; i < padded_size; i++)
	{
		if (center)
		{
			buf[i] = padded_data[i];
		}
		else
		{
			buf[i] = data[i];
		}
	}

	//std::vector<std::vector<std::pair<double, double>>> _fftVal;

	//vect.reserve(framecount);
	int _numFFT = framesize;
	int _numHop = frame_step;
	int _numWindow = -1;
	bool _center = false;
	WindowType _window = HanningWindow;
	PaddingType _pad_mode = E_PADDING_REFLECT;
	librosa_stft(buf, stft, _numFFT, _numHop, _numWindow, _window, _center, _pad_mode);

	//delete buf;
	if (center)	delete padded_data;
}


void FeatureExtractor::istft(std::vector<std::vector<std::pair<double, double>>>& stft, std::vector<double>& pcm_data)
{
	int _numFFT = framesize;
	int _numHop = frame_step;
	int _numWindow = -1;
	int _numLen = -1;
	WindowType _window = HanningWindow;
	bool _center = false;
	pcm_data = librosa_istft(stft, _numHop, _numFFT, _window, _center, _numLen);
}


void FeatureExtractor::concateate_magphase(std::vector<std::vector<double>>& mag_in1, std::vector<std::vector<double>>& mag_in2, std::vector<std::vector<double>>& mag_out)
{
	int _frameCnt = mag_in1.size();
	int i = 0, j = 0;
	mag_out.reserve(_frameCnt);
	for (i = 0; i < _frameCnt; i++)
	{
		int total_size = mag_in1[i].size() + mag_in2[i].size();
		std::vector<double> frameMag;
		frameMag.reserve(total_size);
		for (j = 0; j < mag_in1[i].size(); j++)
		{
			frameMag.push_back(mag_in1[i][j]);
		}
		for (j = 0; j < mag_in2[i].size(); j++)
		{
			frameMag.push_back(mag_in2[i][j]);
		}

		mag_out.push_back(frameMag);
		//mag_out[i].insert(mag_out[i].end(), mag_in2.begin(), mag_in2.end());
	}
}


void FeatureExtractor::librosa_magphase(std::vector<std::vector<std::pair<double, double>>>& mag_out, std::vector<std::vector<std::pair<double, double>>>& fftVal)
{
	int _power = 1;
	int i = 0;
	mag_out.reserve(fftVal.size());
	for (i = 0; i < fftVal.size(); i++)
	{
		mag_out.push_back(magphase(fftVal[i], _power));
	}
}

void FeatureExtractor::librosa_magnitude(std::vector<std::vector<double>>& mag_out, std::vector<std::vector<std::pair<double, double>>>& fftVal)
{
	int _power = 1;
	int i = 0;
	mag_out.reserve(fftVal.size());
	for (i = 0; i < fftVal.size(); i++)
	{
		mag_out.push_back(magnitude(fftVal[i], _power));
	}
}

void FeatureExtractor::librosa_phase(std::vector<std::vector<std::pair<double, double>>>& mag_out, std::vector<std::vector<std::pair<double, double>>>& fftVal)
{
	int i = 0;
	mag_out.reserve(fftVal.size());
	for (i = 0; i < fftVal.size(); i++)
	{
		mag_out.push_back(phase(fftVal[i]));
	}
}


void FeatureExtractor::GetFeature(float * data, int size, std::vector<std::vector<double>>& vect)
{
	Normalization(data, size);
	float *padded_data = Padding(data, size, framesize / 2, REFLECT);
	int padded_size = size + framesize;
	int framecount = (padded_size - framesize) / frame_step + 1;
	int i, j;
	double *buf = new double[framesize];
	
	//TODO: 
	vect.reserve(framecount);

	std::vector<std::vector<double>> mfcc(framecount);
	std::vector<std::vector<double>> delta(framecount);

#ifdef _OPENMP
#pragma omp parallel for private(i,j) schedule(dynamic)
#endif
	for (i = 0; i < framecount; i++)
	{
		for (j = 0; j < framesize; j++)
		{
			buf[j] = padded_data[i * frame_step + j];
		}
		gist->processAudioFrame(buf, framesize);
		mfcc[i] = gist->getMelFrequencyCepstralCoefficients();
	}

	
	Delta(mfcc, delta);

	for (i = 0; i < framecount; i++)
	{
		std::vector<double> feat_vect;
		feat_vect.insert(feat_vect.end(), mfcc[i].begin(), mfcc[i].end());
		feat_vect.insert(feat_vect.end(), delta[i].begin(), delta[i].end());
		vect.push_back(feat_vect);
	}
	
	delete buf;
	delete padded_data;
}



void FeatureExtractor::Delta(const std::vector<std::vector<double>>& in, std::vector<std::vector<double>>& out)
{
	int i, j;
	int cols = (int)in.size();
	int rows = (int)in[0].size();

	std::vector<double> row_vect(cols, 0.0);
	std::vector<double> res;

	if (cols != out.size())
		out.resize(cols);
	for (i = 0; i < cols; i++)
		out[i].reserve(rows);


	for (i = 0; i < rows; i++)
	{
#ifdef _OPENMP
#pragma omp parallel for private(i,j) schedule(static)
#endif
		for (j = 0; j < cols; j++)
		{
			row_vect[j] = in[j][i];
		}
		//TODO. remove hard coding of winsize, deg, delta
		res = sg_derivative(row_vect, 4, 1, 1.0);
		
		for (j = 0; j < cols; j++)
		{
			out[j].push_back(res[j]);
		}
	}
}
