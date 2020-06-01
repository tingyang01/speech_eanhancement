// speaker_separation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "WaveOperation.h"


void speaker_separation()
{
	const char* _szGroundPath = "groundtruth.wav";
	const char* _szMicPath = "mic.wav";
	const char* _szSpeakerPath = "speaker.wav";
	int _numFFT = 768;
	int _numStep = 384;
	int samplerate = 16000;
	FeatureExtractor *featExtractor = new FeatureExtractor(_numFFT, _numStep, samplerate);
	CWaveOperation wavGroundOpt(_szGroundPath, featExtractor);
	CWaveOperation wavMicOpt(_szMicPath, featExtractor);
	CWaveOperation wavSpeakerOpt(_szSpeakerPath, featExtractor);
	int _nLoop = int((wavMicOpt.getAudioSize() - _numFFT) / _numStep);
	int i = 0, j = 0, k = 0;
	std::vector<double> pcmOut;
	pcmOut.reserve(_nLoop*(_numFFT - _numStep));

	for (i = 0; i < _nLoop; i++)
	{
		std::vector<std::vector<std::pair<double, double>>> stftMic;
		std::vector<std::vector<double>> magMic;
		std::vector<std::vector<std::pair<double, double>>> phaseMic;
		std::vector<std::vector<std::pair<double, double>>> stftSpk;
		std::vector<std::vector<double>> magSpk;
		std::vector<std::vector<std::pair<double, double>>> stftGround;
		std::vector<std::vector<double>> magGound;
		std::vector<std::vector<double>> magConcatenate;

		if (!wavMicOpt.getFrameFeature(stftMic, magMic))			// Calculate spectral features from microphone
		{
			printf("invalid mic data");
			break;
		}

		featExtractor->librosa_phase(phaseMic, stftMic);

		if (!wavSpeakerOpt.getFrameFeature(stftSpk, magSpk))		// Calculate spectral features from speaker
		{
			printf("invalid mic data");
			break;
		}

		if (!wavGroundOpt.getFrameFeature(stftGround, magGound))		// Calculate spectral features from clean data
		{
			printf("invalid mic data");
			break;
		}

		std::vector<double> pcm_debug;
		featExtractor->istft(stftMic, pcm_debug);

		// Concatenate microphone magphase and speaker magphase
		featExtractor->concateate_magphase(magMic, magSpk, magConcatenate);

		// Calculate mask
		std::vector<std::pair<double, double>> mask;
		mask.reserve(magGound[0].size());
		for (j = 0; j < magGound[0].size(); j++)
		{
			double magGound0 = magGound[0][j];
			double magSPk0 = magSpk[0][j];
			double magGound1 = magGound[1][j];
			double magSPk1 = magSpk[1][j];
			double mask0 = sqrt(magGound0*magGound0 /(magGound0*magGound0 + magSPk0*magSPk0));
			double mask1 = sqrt(magGound1*magGound1 /(magGound1*magGound1 + magSPk1*magSPk1));
			mask.push_back(std::make_pair(mask0, mask1));
		}

		// Calculate cleaned mic 
		std::vector<std::pair<double, double>> recovered_clean_mag;
		recovered_clean_mag.reserve(mask.size());
		for (j = 0; j < mask.size(); j++)
		{
			double recover_mag0 = magMic[0][j]*mask[j].first;
			double recover_mag1 = magMic[1][j]*mask[j].second;
			recovered_clean_mag.push_back(std::make_pair(recover_mag0, recover_mag1));
		}

		std::vector<std::vector<std::pair<double, double>>> recovered_mag;
		recovered_mag.reserve(phaseMic.size());
		for (j = 0; j < phaseMic.size(); j++)
		{
			//recovered_clean_mag
			std::vector<std::pair<double, double>> channelPhase;
			channelPhase.reserve(phaseMic[j].size());
			for (k = 0; k < phaseMic[j].size(); k++)
			{
				if (j == 0)
				{
					channelPhase.push_back(std::make_pair(phaseMic[j][k].first*std::get<0>(recovered_clean_mag[k]), phaseMic[j][k].second*std::get<0>(recovered_clean_mag[k])));
				}
				else
				{
					channelPhase.push_back(std::make_pair(phaseMic[j][k].first*std::get<1>(recovered_clean_mag[k]), phaseMic[j][k].second*std::get<1>(recovered_clean_mag[k])));
				}
			}
			recovered_mag.push_back(channelPhase);
		}

		std::vector<std::vector<std::pair<double, double>>> stft;
		std::vector<double> pcm_data;
		featExtractor->istft(recovered_mag, pcm_data);
		for (j = _numStep; j < _numFFT; j++)
		{
			pcmOut.push_back(pcm_data[j]);
		}
	}
	wavMicOpt.writeWave("my_separation_result.wav", pcmOut);

	if (featExtractor)
	{
		delete featExtractor;
		featExtractor = NULL;
	}
}


int main()
{
	speaker_separation();
	std::cout << "Hello World!\n";
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
