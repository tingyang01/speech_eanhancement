#pragma once
#include "Gist.h"

enum PadType {
	ZERO,
	REFLECT,
};

class FeatureExtractor
{
public:
	FeatureExtractor(int framesize, int frame_step, int samplerate);
	~FeatureExtractor();
	void GetFeature(float* data, int size, std::vector<std::vector<double>>& feat);
	void stft(float* data, int size, std::vector<std::vector<std::pair<double, double>>>& feat, bool center);
	void istft(std::vector<std::vector<std::pair<double, double>>>& stft, std::vector<double>& pcm_data);
	void concateate_magphase(std::vector<std::vector<double>>& mag_in1, std::vector<std::vector<double>>& mag_in2, std::vector<std::vector<double>>& mag_out);
	void librosa_magphase(std::vector<std::vector<std::pair<double, double>>>& mag_out, std::vector<std::vector<std::pair<double, double>>>& fftVal);
	void librosa_magnitude(std::vector<std::vector<double>>& mag_out, std::vector<std::vector<std::pair<double, double>>>& fftVal);
	void librosa_phase(std::vector<std::vector<std::pair<double, double>>>& mag_out, std::vector<std::vector<std::pair<double, double>>>& fftVal);
	void Delta(const std::vector<std::vector<double>>& in, std::vector<std::vector<double>>& out);
	float* Padding(float* data, int size, int padding_size, PadType pad_type);
	Gist<double> *gist;
	int framesize;
	int samplerate;
	int frame_step;
};

