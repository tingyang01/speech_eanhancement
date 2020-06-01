#pragma once
#include "common.h"
using namespace std;

#define EPS 1e-12

static enum FEATURE_TYPE {
	FEAT_MFCC,
	FEAT_MFCCDELTA,
	FEAT_MFCCDELTADELTA2
} Feature_Type;

template <class T>
class DataArray
{
public:
	DataArray<T>() { };
	~DataArray<T>() {
		if (feature) delete feature;
	};
	DataArray(T*feat, int size)
	{
		this->feature = feat;
		this->size = size;
	}
	T* feature;
	int size;
};


class MFCC
{
public:
	MFCC(int sample_rate, int fftlen, int frame_len, int frame_step, int filt_order, int ceps_order);
	~MFCC();
	void CalcFeatures(float* data, int size, FEATURE_TYPE nType, int deltaWins, vector<vector<double>>& vect);
	double* CalcSegmentMfcc(double* segment, FEATURE_TYPE nType, int deltaWin);
	double* CalcSegmentDelta(double* mfcc, int nOrder, int deltawin);
	double* CalcSegmentDeltaDelta(double* mfcc, int nOrder, int deltadeltawin);
	int GetCepstrumOrder() { return ceps_order; }
	int GetFrameCount() { return framecount; }

	//vad processing
	double* SliceMeanEnergyList(double* data, int size, int vad_win);
	bool IsValidAudioFrame(double *data, int size);

private:
	void Init();
	void CreateFilterBank();
	void Normalization(float* data, int size);
	

private:
	int fftlen;
	int frame_width;
	int frame_step;
	int filter_order;
	int ceps_order;
	int sample_rate;
	int filter_len;
	int low, high;
	int framecount;
	double** filter_bank;
	vector<double> vect_ceps;

	//temp
	double *temp_buf1;
	complex<double>* fftbuffer;
};

