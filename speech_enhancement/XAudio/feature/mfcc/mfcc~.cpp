#include "mfcc.h"
#include "windowing.h"
#include "FFT.h"
#include <algorithm>

#define BUFSIZE 512

// This function initialize filter weights to 0
void init_filterbank(double**w, int num_filt, int filt_len)
{
	int i, j;
	for (i = 0; i < num_filt; i++)
	{
		for (j = 0; j < filt_len; j++)
		{
			w[i][j] = 0.0;
		}
	}
}

// This function creates a Mel weight matrix

void mag_square(complex<double>* vec, double* vec_mag, int FFTLen) // This function computes magnitude squared FFT
{
	int i;
	double temp;
	for (i = 1; i <= FFTLen / 2 + 1; i++)
	{
		temp = vec[i - 1].real()*vec[i - 1].real() + vec[i - 1].imag()*vec[i - 1].imag();
		vec_mag[i - 1] = temp;
	}
}


double frame_energy(double* data, int size)        // This function computes frame energy
{
	int i;
	double frm_en = 0.0;
	for (i = 0; i < size; i++)
	{
		frm_en = frm_en + data[i] * data[i];
	}
	return frm_en / size;
}

// computes log energy of each channel
double* mel_energy(double**w, int num_filt, int len_filt, double* vect_mag)
{
	int i, j;
	double* melEnergy = new double[num_filt];
	for (i = 1; i <= num_filt; i++)    // set initial energy value to 0
		melEnergy[i - 1] = 0.0F;

	for (i = 1; i <= num_filt; i++)
	{
		for (j = 1; j <= len_filt; j++)
			melEnergy[i - 1] = melEnergy[i - 1] + w[i - 1][j - 1] * vect_mag[j - 1];
		melEnergy[i - 1] = (double)(log(melEnergy[i - 1]));
	}
	return melEnergy;
}


// Compute Mel cepstrum
double* cepstrum(double *melEnergy, int ceps_order)
{
	int i, j;
	double *ceps = new double[ceps_order];
	for (i = 0; i < ceps_order; i++)
	{
		ceps[i] = 0.0F;    // initialize to 0
		for (j = 0; j < ceps_order; j++)
			ceps[i] += melEnergy[j] * cos(PI*((double)i) / ((double)ceps_order)*((double)j + 0.5f)); // DCT 2
		ceps[i] = sqrt(2.0 / double(ceps_order))*ceps[i];
	}
	return ceps;
}

template <class T>
bool medfilt1_sort(T a, T b)
{
	return (a < b);
}


template <class T>
void medfilt1(T* s, int size, int dim)
{
	T* medi = (T*)malloc(sizeof(T) * size);
	vector<T> filtdata;
	int window = 0;
	int validNum = 0;
	double INVALID_VALUE = -100000000;

	filtdata.reserve(dim);
	for (int i = 0; i < size; i++)
	{
		filtdata.clear();
		// make filer data
		for (int k = -dim / 2; k <= dim / 2; k++)
		{
			if (i + k < 0 || i + k >= size) continue;
			filtdata.push_back(s[i + k]);
		}
		// Sort in order of big value.
		sort(filtdata.begin(), filtdata.end());
		
		medi[i] = filtdata[dim / 2];
	}
	memcpy(s, medi, sizeof(T) * size);
	free(medi);
}



MFCC::MFCC(int sample_rate, int fftlen, int frame_len, int frame_step, int filt_order, int ceps_order) :
	fftlen(fftlen),
	frame_width(frame_len),
	frame_step(frame_step),
	filter_order(filt_order),
	ceps_order(ceps_order),
	sample_rate(sample_rate)
{
	Init();
}

MFCC::~MFCC()
{
	delete temp_buf1;
	delete fftbuffer;
	if (filter_bank)
	{
		int i;
		for (i = 0; i < filter_order; i++)
			delete filter_bank[i];
		delete filter_bank;
	}
}

double* MFCC::SliceMeanEnergyList(double * data, int size, int vad_win)
{
	int block_cnt = size / vad_win;
	int i, j;
	double* energylist = new double[block_cnt * vad_win];
	for (i = 0; i < block_cnt; i++)
	{
		energylist[i] = 0.0;
		for (j = i * vad_win; j < (i + 1) * vad_win; j++)
		{
			energylist[i] += data[j] * data[j];
		}
		energylist[i] /= vad_win;
	}
	return energylist;
}

bool MFCC::IsValidAudioFrame(double *data, int size)
{
	int i;
	int vad_win = (int)(sample_rate * 0.05);
	double vad_threshold = 0.005;
	int block_cnt = size / vad_win;
	double* energylist = SliceMeanEnergyList(data, size, vad_win);
	int* vadIndecies = new int[block_cnt];
	for (i = 0; i < block_cnt; i++)
	{
		if (energylist[i] < vad_threshold) vadIndecies[i] = 0;
		else vadIndecies[i] = 1;
	}
	medfilt1<int>(vadIndecies, block_cnt, 3);
	int cnt = 0;
	for (i = 0; i < block_cnt; i++)
	{
		if (vadIndecies[i] == 1) cnt++;
	}
	delete vadIndecies;
	delete energylist;
	if (cnt > block_cnt / 2) return true;
	return false;
}

void MFCC::Init()
{
	int i;
	high = sample_rate / 2;
	low = 0;

	filter_len = fftlen / 2 + 1;
	filter_bank = new double*[filter_order];
	for (i = 0; i < filter_order; i++)
		filter_bank[i] = new double[filter_len];

	init_filterbank(filter_bank, filter_order, filter_len);
	CreateFilterBank();

	temp_buf1 = new double[frame_width];
	fftbuffer = new complex<double>[fftlen];
}

void MFCC::CreateFilterBank()
{
	double df = (double)sample_rate / (double)fftlen;    // FFT interval
	int indexlow = (int)round((double)fftlen*(double)low / (double)sample_rate); // FFT index of low freq limit
	int indexhigh = (int)round((double)fftlen*(double)high / (double)sample_rate); // FFT index of high freq limit

	double melmax = 2595.0*log10(1.0 + (double)high / 700.0); // mel high frequency
	double melmin = 2595.0*log10(1.0 + (double)low / 700.0);  // mel low frequency
	double melinc = (melmax - melmin) / (double)(filter_order + 1); //mel half bandwidth

	double *melcenters = new double[filter_order]; // mel center frequencies
	double *fcenters = new double[filter_order];	 // Hertz center frequencies
	int	  *indexcenter = new int[filter_order];	 // FFT index for Hertz centers
	int   *indexstart = new int[filter_order];     //FFT index for the first sample of each filter
	int   *indexstop = new int[filter_order];		 //FFT index for the last sample of each filter

	double increment, decrement; // increment and decrement of the left and right ramp
	double sum = 0.0;
	int i, j;
	for (i = 1; i <= filter_order; i++)
	{
		melcenters[i - 1] = (double)i * melinc + melmin;   // compute mel center frequencies
		fcenters[i - 1] = 700.0*(pow(10.0, melcenters[i - 1] / 2595.0) - 1.0); // compute Hertz center frequencies
		indexcenter[i - 1] = (int)round(fcenters[i - 1] / df); // compute fft index for Hertz centers		 
	}
	for (i = 1; i <= filter_order - 1; i++)  // Compute the start and end FFT index of each channel
	{
		indexstart[i] = indexcenter[i - 1];
		indexstop[i - 1] = indexcenter[i];
	}
	indexstart[0] = indexlow;
	indexstop[filter_order - 1] = indexhigh;
	for (i = 1; i <= filter_order; i++)
	{
		increment = 1.0 / ((double)indexcenter[i - 1] - (double)indexstart[i - 1]); // left ramp
		for (j = indexstart[i - 1]; j <= indexcenter[i - 1]; j++)
			filter_bank[i - 1][j] = ((double)j - (double)indexstart[i - 1])*increment;
		decrement = 1.0 / ((double)indexstop[i - 1] - (double)indexcenter[i - 1]);    // right ramp
		for (j = indexcenter[i - 1]; j <= indexstop[i - 1]; j++)
			filter_bank[i - 1][j] = 1.0 - ((double)j - (double)indexcenter[i - 1])*decrement;
	}

	for (i = 1; i <= filter_order; i++)     // Normalize filter weights by sum
	{
		for (j = 1; j <= fftlen / 2 + 1; j++)
			sum = sum + filter_bank[i - 1][j - 1];
		for (j = 1; j <= fftlen / 2 + 1; j++)
			filter_bank[i - 1][j - 1] = filter_bank[i - 1][j - 1] / sum;
		sum = 0.0;
	}
    delete[] melcenters;
    delete[] fcenters;
    delete[] indexcenter;
    delete[] indexstart;
    delete[] indexstop;
}

void MFCC::Normalization(float* signal, int size)
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


vector<vector<double>>get_time_list(vector<double>vad_energy, double time_step)
{
	vector<vector<double>> segments;
	int frame_cnt = vad_energy.size();
	int* vadIndecies = new int[frame_cnt];
	double tr = 0.0005;
	for (int i = 0; i < frame_cnt; i++)
	{
		vadIndecies[i] = 1;
		if (vad_energy[i] < tr)
		{
			vadIndecies[i] = 0;
		}
	}
	medfilt1<int>(vadIndecies, frame_cnt, 3);
	int st_idx = 0;
	int end_idx = 0;
	int cur_state = -1;

	for (int i = 0; i < frame_cnt; i++)
	{
		if (cur_state != vadIndecies[i])
		{
			if (cur_state != -1 && cur_state == 1)
			{
				// check if current segment is not silence
				// add segment
				vector<double> sub_seg;
				sub_seg.push_back(st_idx*time_step);// start time
				sub_seg.push_back(end_idx*time_step);// endtime time
				segments.push_back(sub_seg);
			}

			st_idx = i;
			end_idx = i;
		}
		else
		{
			end_idx = i;
		}
		cur_state = vadIndecies[i];
	}
	delete vadIndecies;
	return segments;
}

void MFCC::CalcFeatures(float* data, int size, FEATURE_TYPE nType, int deltaWins, vector<vector<double>>& vect)
{
	framecount = (size - frame_width) / frame_step + 1;
	int i, j;
	vector<double> feat;
	double tr = 0.0005;
	// vector<double> vad_energy(framecount);
	double *buf = new double[frame_width];
	Normalization(data, size);
	for (i = 0; i < framecount; i++)
	{
		for (j = 0; j < frame_width; j++)
		{
			buf[j] = (double)data[i * frame_step + j];
		}
		double energy = frame_energy(buf, frame_width);
		// vad_energy[i] = energy;
		// printf("frame energy ------> %.10lf\n", energy);
		// if (!IsValidAudioFrame(buf, frame_width)) continue;   // vad processing.
		if (energy < tr) continue;
		double* segfeat = CalcSegmentMfcc(buf, nType, deltaWins);
		vect.push_back({ segfeat,  segfeat + (ceps_order -1) * ((int)nType + 1) });
		delete segfeat;
	}
	// vector<vector<double>>segments = get_time_list(vad_energy, 0.01);
	delete buf;
}

double* MFCC::CalcSegmentMfcc(double* data, FEATURE_TYPE nType, int deltaWin)
{
	double* fft_mag = new double[fftlen / 2 + 1];
	double energy;
	double* melEnergy;
	double* mfcc = NULL;
	double* delta = NULL;
	double* delta2 = NULL;
	double* feat = NULL;

	//hamming(data, frame_width, temp_buf1);
	hanning(data, frame_width, temp_buf1);
	energy = frame_energy(data, frame_width);
	zero_fft(temp_buf1, frame_width, fftlen, fftbuffer);
	mag_square(fftbuffer, fft_mag, fftlen);
	melEnergy = mel_energy(filter_bank, filter_order, filter_len, fft_mag);
	mfcc = cepstrum(melEnergy, ceps_order);
	mfcc[0] = log(energy + EPS);
	double* mfcc_1 = mfcc + 1;
	int ceps_order_1 = ceps_order - 1;
	switch (nType)
	{
	case FEAT_MFCC:
		feat = new double[ceps_order_1];
		memcpy(feat, mfcc_1, sizeof(double) * ceps_order_1);
		break;
	case FEAT_MFCCDELTA:
		feat = new double[ceps_order_1 * 2];
		delta = CalcSegmentDelta(mfcc_1, ceps_order_1, deltaWin);
		memcpy(feat, mfcc_1, sizeof(double) * ceps_order_1);
		memcpy(feat + ceps_order_1, delta, sizeof(double) * ceps_order_1);
		break;
	case FEAT_MFCCDELTADELTA2:
		feat = new double[ceps_order_1 * 3];
		delta = CalcSegmentDelta(mfcc_1, ceps_order_1, deltaWin);
		delta2 = CalcSegmentDeltaDelta(delta, ceps_order_1, deltaWin);
		memcpy(feat, mfcc_1, sizeof(double) * ceps_order_1);
		memcpy(feat + ceps_order_1, delta, sizeof(double) * ceps_order_1);
		memcpy(feat + ceps_order_1 * 2, delta2, sizeof(double) * ceps_order_1);
		break;
	}
	delete melEnergy;
	delete fft_mag;
	if (mfcc) delete mfcc;
	if (delta) delete delta;
	if (delta2) delete delta2;
	return feat;
}

double* MFCC::CalcSegmentDelta(double* mfcc,int nOrder, int deltawin)
{
	int M, i, j, idx1, idx2;
	double temp1, temp2;
	if (deltawin != 2 && deltawin % 2 == 0)
	{
		return NULL;
	}
	if (deltawin > nOrder) return NULL;

	double* delta = new double[nOrder];
	if (deltawin == 2)
	{
		delta[0] = mfcc[0];
		for (i = 1; i < nOrder; i++)
		{
			delta[i] = mfcc[i] - mfcc[i - 1];
		}
	}
	else
	{
		M = deltawin / 2;
		temp2 = 0;
		for (i = -M; i <= M; i++) temp2 += i * i;
		for (i = 0; i < nOrder; i++)
		{
			temp1 = 0;
			for (j = 1; j <= M; j++)
			{
				idx1 = i - j;
				idx2 = i + j;
				if (idx1 < 0) idx1 = 0;
				if (idx2 > nOrder - 1) idx2 = nOrder - 1;
				temp1 += (mfcc[idx2] - mfcc[idx1]) * j;
			}
			delta[i] = temp1 / (double)temp2;
		}
	}
	
	return delta;
}

double* MFCC::CalcSegmentDeltaDelta(double* delta, int nOrder, int win)
{
	return CalcSegmentDelta(delta, nOrder, win);
}


