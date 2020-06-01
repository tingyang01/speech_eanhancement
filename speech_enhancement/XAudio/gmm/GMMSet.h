#pragma once
#include "GaussianMixtureModel.h"

class GMMSet
{
public:
	GMMSet();
	GMMSet(int gmm_order, double reject_threshold);
	~GMMSet();

	void Fit(std::vector<std::vector<double>>& X, std::string label);
	double ScoreInstance_one(int idx, std::vector<double>&);
	double* ScoreAll_one(std::vector<double>&);
	double ScoreInstance(int idx, std::vector<std::vector<double>>& X);
	double* ScoreAll(std::vector<std::vector<double>>& X);

	std::string Predict_one(std::vector<double>&);
	std::string Predict(std::vector<std::vector<double>>& X);
	double GetPredictScore(std::vector<std::vector<double>>& wav_data, std::string label);

	void Load(std::vector<std::string> model_paths);
	void Load(std::string fname);
	void Dump();
	void Dump(std::string fname);

	template <class T>
	int Exist(std::vector<T> vect, T val);

	std::vector<GaussianMixtureModel*> mGmms;
	int mMixtures;
	double mRejectThreshold;
	std::vector<std::string> mLabels;
	int covariance_type;
	GMMParameter params;
};

template<class T>
inline int GMMSet::Exist(std::vector<T> vect, T val)
{
	int i;
	int cnt = vect.size();
	for (i = 0; i < cnt; i++) {
		if (vect[i] == val)
			return i;
	}
	return -1;
}
