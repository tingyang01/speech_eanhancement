#include "CWave.h"
#ifdef _USE_LIBSND
#include "sndfile.h"
#else
#include "wave/file.h"
#endif
#include <stdlib.h>
#include <memory.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include "AudioFile.h"
#define BUFSIZE 512


#ifdef _USE_LIBSND
short int* wav_read_short(const char* path)
{
	SF_INFO sfInfo;
	SNDFILE *pfile;
	int frameCount, channels, totalReadCount;
	short int* buf1, *buf2;
	short int* buffer;
	
	if (!(pfile = sf_open(path, SFM_READ, &sfInfo)))
	{
		printf("Error : could not open \"%s\.\n", path);
		puts(sf_strerror(NULL));
		return NULL;
	}
	frameCount = (int)sfInfo.frames;
	channels = (int)sfInfo.channels;
	buffer = new short int[frameCount * channels];
	buf1 = new short int[BUFSIZE * channels];
	buf2 = new short int[BUFSIZE];
	totalReadCount = 0;
	while (totalReadCount < frameCount)
	{
		int reads = (int)sf_readf_short(pfile, buf1, BUFSIZE);
		wav2mono_short(buf1, channels, BUFSIZE, buf2);
		memcpy(buffer + totalReadCount, buf2, BUFSIZE);
		totalReadCount += reads;
	}
	delete buf1;
	delete buf2;
	sf_close(pfile);
	return buffer;
}


AudioData* wav_read_float(const char* path)
{
	SF_INFO sfInfo;
	SNDFILE *pfile;
	int frameCount, samplerate, channels, totalReadCount;
	float* buf1, *buf2;
	float* buffer;

	if (!(pfile = sf_open(path, SFM_READ, &sfInfo)))
	{
		printf("Error : could not open \"%s\.\n", path);
		puts(sf_strerror(NULL));
		return NULL;
	}
	frameCount = (int)sfInfo.frames;
	channels = (int)sfInfo.channels;
	samplerate = sfInfo.samplerate;
	buffer = new float[frameCount];
	buf1 = new float[BUFSIZE * channels];
	buf2 = new float[BUFSIZE];
	totalReadCount = 0;
	while (totalReadCount < frameCount)
	{
		int reads = (int)sf_readf_float(pfile, buf1, BUFSIZE);
		wav2mono(buf1, channels, reads, buf2);
		memcpy(buffer + totalReadCount, buf2, reads * 4);
		totalReadCount += reads;
	}
	delete buf1;
	delete buf2;
	sf_close(pfile);
	return new AudioData(buffer, totalReadCount, samplerate);
}



#else
AudioData* wav_read_float(const char* path)
{
	int samplerate;
	float* buffer;
	wave::File read_file;
	wave::Error err = read_file.Open(path, wave::kIn);
	if (err) {
		printf("Something went wrong in in open\n");
		return nullptr;
	}
	samplerate = read_file.sample_rate();
	int bit_per_sample = read_file.bits_per_sample();
	int channel_number = read_file.channel_number();

	std::vector<float> content;
	err = read_file.Read(&content);
	if (err) {
		printf("Something went wrong in in read\n");
		return nullptr;
	}
	int i = 0;
	int cnt = content.size();
	buffer = new float[cnt];
	std::copy(content.begin(), content.end(), buffer);
	return new AudioData(buffer, cnt, samplerate);
}

#endif

// WAVE PCM soundfile format (you can find more in https://ccrma.stanford.edu/courses/422/projects/WaveFormat/ )
typedef struct header_file
{
	char chunk_id[4];
	int chunk_size;
	char format[4];
	char subchunk1_id[4];
	int subchunk1_size;
	short int audio_format;
	short int num_channels;
	int sample_rate;			// sample_rate denotes the sampling rate.
	int byte_rate;
	short int block_align;
	short int bits_per_sample;
	char subchunk2_id[4];
	int subchunk2_size;			// subchunk2_size denotes the number of samples.
} header;

typedef struct header_file* header_p;

void wave_write_float(const char* path, double* pcmData, int pcmLen, int sampleRate)
{
	AudioFile<double>::AudioBuffer buffer;
	AudioFile<double> audioFile;
	buffer.resize(1);
	buffer[0].resize(pcmLen);
	int _numChannel = 1;
	int numSamplesPerChannel = pcmLen;
	for (int i = 0; i < pcmLen; i++)
	{
		buffer[0][i] = pcmData[i];
	}
	bool ok  = audioFile.setAudioBuffer(buffer);
	audioFile.setBitDepth(16);
	audioFile.setSampleRate(sampleRate);
	//audioFile.setNumChannels(1);
	//audioFile.setNumSamplesPerChannel(numSamplesPerChannel);
	audioFile.save(path);
}


void wave_write_float_1(const char* path, double* pcmData, int pcmLen, int sampleRate)
{
	wave::File write_file;
	wave::Error err = write_file.Open(path, wave::kOut);
	if (err) {
		std::cout << "Something went wrong in out open" << std::endl;
		//return 3;
		return;
	}
	int bit_per_sample = 32;
	int channel_number= 1;
	std::vector<float> content;
	content.resize(pcmLen);
	for (int i = 0; i < pcmLen; i++)
	{
		content[i] = pcmData[i];
	}
	write_file.set_sample_rate(sampleRate);
	write_file.set_bits_per_sample(bit_per_sample);
	write_file.set_channel_number(channel_number);
	err = write_file.Write(content);
	if (err) {
		std::cout << "Something went wrong in write" << std::endl;
		//return 4;
	}

}

void wav2mono_short(short int* buf, int channelCount, int buf_size, short int *dst)
{
	if (channelCount == 1)
	{
		memcpy(dst, buf, sizeof(short int) * buf_size);
	}
	else
	{
		int i, j;
		int temp;
		for (i = 0; i < buf_size; i++)
		{
			temp = 0;
			for (j = 0; j < channelCount; j++)
				temp += (int)buf[i * channelCount + j];
			dst[i] = (short int)(temp / channelCount);
		}
	}
}

void wav2mono(float* buf, int channelCount, int buf_size, float *dst)
{
	if (channelCount == 1)
	{
		memcpy(dst, buf, sizeof(float) * buf_size);
	}
	else
	{
		int i, j;
		float temp;
		for (i = 0; i < buf_size; i++)
		{
			temp = 0.0;
			for (j = 0; j < channelCount; j++)
				temp += buf[i * channelCount + j];
			dst[i] = temp / (float)channelCount;
		}
	}
}