#include "WaveOperation.h"


CWaveOperation::CWaveOperation(const char * wavPath, FeatureExtractor* _featExtractor)
{
	m_pcmAudio = wav_read_float(wavPath);
	m_featExtractor = _featExtractor;
	m_nCurIdx = 0;
	m_nNFFT = m_featExtractor->framesize;
	m_nHOPE_LEN = m_featExtractor->frame_step;
	m_pCurFrameData = new float[3 * m_nHOPE_LEN];
	memset(m_pCurFrameData, 0, sizeof(float) * 3 * m_nHOPE_LEN);
}


CWaveOperation::~CWaveOperation()
{
	if (m_pcmAudio != NULL)
	{
		delete m_pcmAudio;
		m_pcmAudio = NULL;
	}

	if (m_pCurFrameData) {
		delete m_pCurFrameData;
		m_pCurFrameData = NULL;
	}
}

int CWaveOperation::getAudioSize()
{
	if (m_pcmAudio)
	{
		return m_pcmAudio->size;
	}
	return 0;
}

bool CWaveOperation::isValidSp()
{
	return m_pcmAudio != NULL && m_pcmAudio->size - m_nHOPE_LEN > m_nCurIdx;
}

bool CWaveOperation::getFrameFeature(std::vector<std::vector<std::pair<double, double>>>& stft, std::vector<std::vector<double>>& mag_out)
{
	//loat* data = m_pcmAudio->data[];
	// operate current frame buffer
	int i = 0;
	if (!isValidSp()) {
		return false;
	}

	// data push at front 
	for (i = 0; i < m_nHOPE_LEN; i++)
	{
		m_pCurFrameData[i] = m_pCurFrameData[i + m_nNFFT];
	}

	// set concatnate new input to current frame data from HOP_LEN
	for (i = 0; i < m_nNFFT; i++)
	{
		m_pCurFrameData[i + m_nHOPE_LEN] = m_pcmAudio->data[m_nCurIdx + i];
	}

	// std::vector<std::vector<std::pair<double, double>>> _stft;
	// std::vector<std::vector<std::pair<double, double>>>& mag_out
	bool isCenter = false;
	// short term FFT
	m_featExtractor->stft(m_pCurFrameData, m_nHOPE_LEN * 3, stft, isCenter);

	// MagPhase
	m_featExtractor->librosa_magnitude(mag_out, stft);

	// Increase current index
	m_nCurIdx += m_nHOPE_LEN;
	return true;
}

void CWaveOperation::writeWave(const char * wavFile, std::vector<double>& wav_data)
{
	double* _pData = &wav_data[0];
	wave_write_float(wavFile, _pData, wav_data.size(), m_pcmAudio->samplerate);
}

