#include "spectrum.h"
#include "../fft/kiss_fft130/kiss_fft.h"

void librosa_stft(
	std::vector<double> in, 
	std::vector<std::vector<std::pair<double, double>>>& out, 
	int n_fft, 
	int hop_length, 
	int win_length,
	WindowType windowType, 
	bool center, 
	PaddingType pad_mode)
{
	int window_len = win_length;
	int hop_len = hop_length;

	if (win_length == -1) {
		window_len = n_fft;
	}

	if (hop_length == -1) {
		hop_len = win_length / 4;
	}

	kiss_fft_cfg cfg;     /**< Kiss FFT configuration */
	kiss_fft_cpx* fftIn;  /**< FFT input samples, in complex form */
	kiss_fft_cpx* fftOut; /**< FFT output samples, in complex form */

	fftIn = new kiss_fft_cpx[n_fft];
	fftOut = new kiss_fft_cpx[n_fft];
	cfg = kiss_fft_alloc(n_fft, 0, 0, 0);

	std::vector<double> windowFunction;    /**< The window function used in FFT processing */
	
	int i;

	bool fftBins = true;
	windowFunction = WindowFunctions<double>::createWindow(window_len, windowType, fftBins);

	if (center) {
		//TODO padding
	}


	std::vector<std::vector<double>> frames = MakeFrame(in, n_fft, hop_len);

	int frame_count = frames.size();

	for (i = 0; i < frame_count; i++) {
		std::vector<std::pair<double, double>> fft_vector;
		for (int j = 0; j < n_fft; j++)
		{
			fftIn[j].r = frames[i][j] * windowFunction[j];
			fftIn[j].i = 0.0;
		}
		// execute kiss fft
		kiss_fft(cfg, fftIn, fftOut);

		// store real and imaginary parts of FFT
		for (int j = 0; j < n_fft / 2 + 1; j++)
		{
			fft_vector.push_back(std::pair<double, double>(fftOut[j].r, fftOut[j].i));
		}

		out.push_back(fft_vector);
	}

	kiss_fft_free(fftIn);
	kiss_fft_free(fftOut);
	kiss_fft_free(cfg);
}


std::vector<double> librosa_istft(
	std::vector<std::vector<std::pair<double, double>>> stft_matrix,
	int hop_length,
	int win_length,
	WindowType windowType,
	bool center,
	int length
) {
	int n_frames = stft_matrix.size();
	//KCI 
	//TODO  check valid the input value
	//in this case input(stft_matrix) is not empty vector, so that input should have one element at least.
	int frame_len = stft_matrix[0].size();
	int n_fft = 2 * (frame_len - 1);
	int window_len = win_length;
	int hop_len = hop_length;

	if (window_len == -1) {
		window_len = n_fft;
	}

	if (hop_len == -1) {
		hop_len = win_length / 4;
	}

	kiss_fft_cfg cfg;     /**< Kiss FFT configuration */
	kiss_fft_cpx* fftIn;  /**< FFT input samples, in complex form */
	kiss_fft_cpx* fftOut; /**< FFT output samples, in complex form */

	fftIn = new kiss_fft_cpx[n_fft];
	fftOut = new kiss_fft_cpx[n_fft];
	cfg = kiss_fft_alloc(n_fft, 1, 0, 0);

	std::vector<double> windowFunction;    /**< The window function used in IFFT processing */
	bool fftBins = true;
	windowFunction = WindowFunctions<double>::createWindow(window_len, windowType, fftBins);

	windowFunction = PaddingCenter(windowFunction, n_fft);

	if (length != -1) {
		//TODO
		//
	}

	int expected_signal_len = n_fft + hop_length * (n_frames - 1);
	
	//std::vector<double> out(expected_signal_len);
	double * y = new double[expected_signal_len];
	memset(y, 0, sizeof(double) * expected_signal_len);

	int i, j;
	for (i = 0; i < n_frames; i++) {

		for (j = 0; j < n_fft; j++) {
			if (j < frame_len) {
				fftIn[j].r = stft_matrix[i][j].first;
				fftIn[j].i = stft_matrix[i][j].second;
			}
			else {
				fftIn[j].r = 0;
				fftIn[j].i = 0;
			}
		}

		kiss_fft(cfg, fftIn, fftOut);

		for (j = 0; j < n_fft; j++) {
			y[hop_len * i + j] += windowFunction[j] * fftOut[j].r/ n_fft;
		}
		
		//y[hop_len * i]
	}

	std::vector<double> ifft_window_sum = window_sumsquare(windowType, n_frames, hop_len, window_len, n_fft);

	double tiny = 1.0 / INFINITY;// 1e-307;

	for (i = 0; i < expected_signal_len; i++) {
		if (ifft_window_sum[i] > tiny) 
			y[i] /= ifft_window_sum[i];
	}

	if (length == -1) {
		if (center) {
			//y = y[int(n_fft // 2):-int(n_fft // 2)]
			return std::vector<double>(y + n_fft / 2, y + expected_signal_len - n_fft / 2);
		}
		return std::vector<double>(y, y+ expected_signal_len);
	}
	else {
		int start = 0;
		if (center) {
			start = n_fft / 2;
		}
		return std::vector<double>(y + start, y + start + length);
	}
	kiss_fft_free(fftIn);
	kiss_fft_free(fftOut);
	kiss_fft_free(cfg);
}

std::vector<std::pair<double, double>> magphase(std::vector<std::pair<double, double>> in, int power) {
	int size = in.size();
	std::vector<std::pair<double, double>> out;
	out.reserve(size);
	for (int i = 0; i < size; i++) {
		out.push_back(std::pair<double, double>(pow(sqrt(in[i].first * in[i].first + in[i].second * in[i].second), power) ,
			atan(in[i].second / in[i].first)));
	}
	// * 180 / M_PI
	return out;
}


std::vector<double> magnitude(std::vector<std::pair<double, double>> in, int power) {
	int size = in.size();
	std::vector<double> out;
	out.reserve(size);
	for (int i = 0; i < size; i++) {
		double mag = pow(sqrt(in[i].first*in[i].first + in[i].second*in[i].second), power);
		out.push_back(mag);
	}
	return out;
}

std::vector<std::pair<double, double>> phase(std::vector<std::pair<double, double>> in) {
	int size = in.size();
	std::vector<std::pair<double, double>> out;

	for (int i = 0; i < size; i++) {
		double angle = atan2(in[i].second, in[i].first);
		out.push_back(std::pair<double, double>(cos(angle), sin(angle)));
	}
	return out;
}

std::vector<std::vector<double>> MakeFrame(std::vector<double> in, int frame_len, int hop_len) {

	std::vector<std::vector<double>> out;
	int input_len = in.size();

	//KCI
	//TODO if input_len = frame_len + hop_len * m or not
	//in this case input_len = frame_len + hop_len * m

	int frame_count = (input_len - frame_len + 1) / hop_len + 1;

	for (int i = 0; i < frame_count; i++) {
		std::vector<double> frame;
		frame.reserve(frame_len);
		for (int j = 0; j < frame_len; j++) {
			frame.push_back(in.at(i * hop_len + j));
		}
		out.push_back(frame);
	}
	return out;
}

std::vector<double> PaddingCenter(std::vector<double> in, int padded_size)
{
	std::vector<double> out;
	int left_pad = (padded_size - in.size()) / 2;
	int in_size = in.size();

	for (int i = 0; i < padded_size; i++) {
		if (i < left_pad) out.push_back(0);
		else if (i < left_pad + in_size) out.push_back(in.at(i - left_pad));
		else out.push_back(0);
	}
	return out;
}


std::vector<double> window_sumsquare(WindowType windowType, int n_frames, int hop_length, int win_length, int n_fft,
	int norm) {
	int window_len = win_length;
	if (window_len == -1) {
		window_len = n_fft;
	}
	bool fftBins = true;
	std::vector<double> win_sq = WindowFunctions<double>::createWindow(window_len, windowType, fftBins);

	int n = n_fft + hop_length * (n_frames - 1);
	double *x = new double[n];
	memset(x, 0, sizeof(double) * n);

	int i;
	//normalize
	//TODO
	//in this case no use normalize

	/*double sq = 0;
	for (i = 0; i < window_len; i++) {
		sq += win_sq[i] * win_sq[i];
	}
	sq = sqrt(sq);
	
	double maxmag = 0.0;
	for (i = 0; i < window_len; i++) {
		if (maxmag < abs(win_sq[i]))
			maxmag = abs(win_sq[i]);
	}

	double tiny = 1e-308;

	for (int i = 0; i < window_len; i++) {
		if (win_sq[i] > tiny) {
		}
	}*/
	// win_sq ** 2
	for (i = 0; i < win_sq.size(); i++) {
		win_sq.at(i) = win_sq[i] * win_sq[i];
	}
	std::vector<double> win_sq_padded = PaddingCenter(win_sq, n_fft);
	__window_ss_fill(x, n,  win_sq_padded, n_frames, hop_length);

	std::vector<double> out(x, x+n);
	delete x;
	return out;
}

void __window_ss_fill(double *x, int n,  std::vector<double> win_sq, int n_frames, int hop_length) {
	int n_fft = win_sq.size();
	for (int i = 0; i < n_frames; i++) {
		for (int j = 0; j < n_fft; j++) {
			x[i * hop_length + j] += win_sq[j];
		}
	}
}
