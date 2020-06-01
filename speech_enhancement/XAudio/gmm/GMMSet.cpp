#include <fstream>
#include "GMMSet.h"


GMMSet::GMMSet()
{
	mMixtures = 64;
	mRejectThreshold = 10;
	params.nr_mixture = mMixtures;
	covariance_type = COVTYPE_DIAGONAL;
	params.min_covar = 1e-3;
	params.threshold = 0.01;
	params.nr_iteration = 300;
	params.init_with_kmeans = 0;
	params.concurrency = 2; // cpu_count();
	params.verbosity = 0;
}

GMMSet::GMMSet(int mixtures, double reject_threshold) {
	this->mMixtures = mixtures;
	this->mRejectThreshold = reject_threshold;
}

GMMSet::~GMMSet()
{
	mGmms.clear();
	mLabels.clear();
}

void GMMSet::Fit(std::vector<std::vector<double>>& X, std::string label) {
	mLabels.push_back(label);
	GaussianMixtureModel* gmm = new GaussianMixtureModel(mMixtures, COVTYPE_DIAGONAL);
	params.nr_instance = (int)X.size();
	params.nr_dim = (int)X[0].size();
	gmm->TrainModel(X, &params);
	mGmms.push_back(gmm);	
}

double GMMSet::ScoreInstance_one(int idx, std::vector<double>& x)
{
	return mGmms[idx]->ScoreInstance(x);
}

double* GMMSet::ScoreAll_one(std::vector<double>& x)
{
	int i;
	int cnt = (int)mGmms.size();
	double *res = new double[cnt];
	for (i = 0; i < cnt; i++)
		res[i] = ScoreInstance_one(i, x);
	return res;
}

double GMMSet::ScoreInstance(int idx, std::vector<std::vector<double>>& X)
{
	return mGmms[idx]->ScoreAll(X, params.concurrency);
}

double* GMMSet::ScoreAll(std::vector<std::vector<double>>& X)
{
	int i;
	int cnt = (int)mGmms.size();
	double *res = new double[cnt];
	for (i = 0; i < cnt; i++)
		res[i] = ScoreInstance(i, X);
	return res;
}


std::string GMMSet::Predict_one(std::vector<double>& x)
{
	double* scores = ScoreAll_one(x);
	int res = 0;
	int i;
	int gmm_cnt = (int)mGmms.size();
	double maxScore = scores[0];
	for (i = 1; i < gmm_cnt; i++)
	{
		if (maxScore < scores[i]) {
			maxScore = scores[i];
			res = i;
		}
	}
	return mLabels[res];
}

std::string GMMSet::Predict(std::vector<std::vector<double>>& X)
{
	double* scores = ScoreAll(X);
	int res = 0;
	int i;
	int gmm_cnt = (int)mGmms.size();

	double maxscore = scores[0];
	for (i = 1; i < gmm_cnt; i++)
	{
		if (maxscore < scores[i]) {
			maxscore = scores[i];
			res = i;
		}
	}
	return mLabels[res];
}

double GMMSet::GetPredictScore(std::vector<std::vector<double>>& wav_data, std::string label)
{
	int idx = Exist(mLabels, label);
	if (idx < 0) {
		printf("There is no label named %s\n", label);
		return 0.0;
	}
	if (wav_data.size() <= 0)
	{
		printf("File length is too short.\n");
		return 0.0;
	}
	return ScoreInstance(idx, wav_data) / wav_data.size();
}

void GMMSet::Load(std::vector<std::string> model_paths)
{
	int i;
	int cnt = (int)model_paths.size();
	mGmms.reserve(cnt);
	for (i = 0; i < cnt; i++)
	{
		GaussianMixtureModel *gmm = new GaussianMixtureModel(model_paths[i]);
		mGmms.push_back(gmm);
		int st = model_paths[i].find("/") + 1;
		int ed = model_paths[i].find(".gmdl");
		std::string label = model_paths[i].substr(st, ed - st);
		mLabels.push_back(label);
	}
}

void GMMSet::Dump()
{
	int cnt = (int)mGmms.size();
	int i;
	for (i = 0; i < cnt; i++)
	{
		std::string gmm_path;
		gmm_path = "model\\" + mLabels[i] + ".gmdl";
		std::ofstream fout(gmm_path);
		mGmms[i]->Dump(fout);
		fout.close();
	}
}

void GMMSet::Dump(std::string fname) 
{
	int i;
	size_t cnt = mGmms.size();
	std::ofstream fout(fname);

	//nr_mixture, covariance_type
	fout << mMixtures << std::endl;
	fout << covariance_type << std::endl;

	//mGmms
	fout << cnt << std::endl;
	for (i = 0; i < cnt; i++)
	{
		fout << mLabels[i] << std::endl;
		mGmms[i]->Dump(fout);
	}
	fout.close();
}

void GMMSet::Load(std::string fname)
{
	std::ifstream fin(fname);
	int nr_mixtures, cov_type, cnt, i;
	fin >> nr_mixtures >> cov_type;
	fin >> cnt;
	for (i = 0; i < cnt; i++)
	{
		std::string label;
		fin >> label;
		mLabels.push_back(label);
		GaussianMixtureModel *gmm = new GaussianMixtureModel(nr_mixtures, covariance_type);
		gmm->Load(fin);
		mGmms.push_back(gmm);
	}
	fin.close();
}