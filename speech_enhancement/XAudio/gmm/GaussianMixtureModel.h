#pragma once
#include "core/gmm.hh"

typedef std::vector<std::vector<real_t>> DenseDataset;

struct GMMParameter {
	int nr_instance;
	int nr_dim;
	int nr_mixture;
	double min_covar;
	double threshold;
	int nr_iteration;
	int init_with_kmeans;
	int concurrency;
	int verbosity;
};

class GaussianMixtureModel : private GMM
{
public:
	GaussianMixtureModel(int nr_mixture, int covariance_type);
	GaussianMixtureModel(std::string path);
	~GaussianMixtureModel();

	void TrainModel(std::vector<std::vector<double>> &X_in, GMMParameter *param);
	void TrainModelFromUBM(GMM *ubm, double **X_in, GMMParameter *param);
	void Dump(std::string model_file);
	void Dump(std::ofstream& fout);
	void Load(std::string model_file);
	void Load(std::ifstream& fin);

	double ScoreAll(std::vector<std::vector<double>> X_in, int concurrency);
	void ScoreBatch(double **X_in, double *prob_out, int nr_instance, int nr_dim, int concurrency);
	double ScoreInstance(std::vector<double>&);

	int GetDim() { return dim; }
	int GetMixtures() { return nr_mixtures; }

	void PrintParam(GMMParameter *param);
	void Print_X(double **X);

private:
	void conv_double_pp_to_vv(double **Xp, DenseDataset &X, int nr_instance, int nr_dim);
	void conv_double_p_to_v(double *x_in, std::vector<real_t> &x, int nr_dim);
};

