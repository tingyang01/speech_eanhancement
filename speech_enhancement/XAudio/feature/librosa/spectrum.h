#pragma once
#include "../window/WindowFunctions.h"

enum PaddingType
{
	E_PADDING_REFLECT
};

void librosa_stft(std::vector<double> in, std::vector<std::vector<std::pair<double, double>>> &out, int n_fft = 2048, int hop_length = -1, 
	int win_length = -1, WindowType window = HanningWindow,	bool center = true, PaddingType pad_mode = E_PADDING_REFLECT);
std::vector<double> librosa_istft(
	std::vector<std::vector<std::pair<double, double>>> stft_matrix,
	int hop_length = -1,
	int win_length = -1,
	WindowType windowType = HanningWindow,
	bool center = true,
	int length = -1
);


std::vector<std::pair<double, double>> magphase(std::vector<std::pair<double, double>> in, int power = 1);

std::vector<double> magnitude(std::vector<std::pair<double, double>> in, int power = 1);

std::vector<std::pair<double, double>> phase(std::vector<std::pair<double, double>> in);

std::vector<std::vector<double>> MakeFrame(std::vector<double> in, int frame_len, int hop_len);

std::vector<double> PaddingCenter(std::vector<double> in, int padded_size);

std::vector<double> window_sumsquare(WindowType windowType, int n_frames, int hop_length = 512, int win_length = -1, int n_fft = 2048,
	int norm = -1);

void __window_ss_fill(double *x, int n, std::vector<double> win_sq, int n_frames, int hop_length);