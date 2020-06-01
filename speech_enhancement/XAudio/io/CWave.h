#pragma once

#ifdef __cplusplus
extern "C"{
#endif
class AudioData
{
public:
	AudioData(float * data, int size, int samplerate) :
		data(data),
		size(size),
		samplerate(samplerate)
	{

	}
	~AudioData() {
		delete data;
		data = nullptr;
	}

	float* data;
	int samplerate;
	int size;
};

short int* wav_read_short(const char* path);
AudioData* wav_read_float(const char* path);
void write_short(const char* path, short int* data);
void write_float(const char* path, float* data);
void wave_write_float(const char* path, double* pcmData, int pcmLen, int sampleRate);

void wav2mono_short(short int* src, int channels, int len, short int* dst);
void wav2mono(float* src, int channels, int len, float* dst);
#ifdef __cplusplus
}
#endif
