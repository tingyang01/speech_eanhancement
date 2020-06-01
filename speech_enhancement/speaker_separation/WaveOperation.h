#pragma once
#include "gmm/GMMSet.h"
#include "io/CWave.h"
#include "feature/FeatureExtractor.h"

class CWaveOperation
{
public:
	CWaveOperation(const char* wavPath, FeatureExtractor* _featExtractor);
	~CWaveOperation();

public:
	AudioData* m_pcmAudio;
	FeatureExtractor* m_featExtractor;
	std::vector<std::vector<std::vector<double>>> features;

public:
	int getAudioSize();					// get Audio size
	bool isValidSp();
	bool getFrameFeature(std::vector<std::vector<std::pair<double, double>>>& stft, 
						std::vector<std::vector<double>>& mag_out);				// Get frame feature data
	void writeWave(const char* wavFile, std::vector<double>& wav_data);
private:
	int m_nCurIdx;						// Current segment index
	int m_nNFFT;						// FFT number(768)
	int m_nHOPE_LEN;					// Hope length(384)
	float* m_pCurFrameData;
};


