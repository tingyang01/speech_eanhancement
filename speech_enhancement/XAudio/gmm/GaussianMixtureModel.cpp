#include "GaussianMixtureModel.h"
#include <fstream>

GaussianMixtureModel::GaussianMixtureModel(int nr_mixture, int covariance_type)
	:GMM(nr_mixture, covariance_type)
{
}

GaussianMixtureModel::GaussianMixtureModel(std::string path)
	:GMM(path)
{
}


GaussianMixtureModel::~GaussianMixtureModel()
{
}

void GaussianMixtureModel::TrainModel(std::vector<std::vector<double>>& X_in, GMMParameter * param)
{
	PrintParam(param);
	GMMTrainerBaseline tr(param->nr_iteration, param->min_covar,
		param->threshold, param->init_with_kmeans, param->concurrency, param->verbosity);
	trainer = &tr;
	fit(X_in);
	trainer = NULL;
}

void GaussianMixtureModel::TrainModelFromUBM(GMM * ubm, double ** X_in, GMMParameter * param)
{
	PrintParam(param);
	trainer = new GMMUBMTrainerBaseline(ubm, param->nr_iteration, param->min_covar,
		param->threshold, param->concurrency, param->verbosity);
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, param->nr_instance, param->nr_dim);
	fit(X);
	trainer = NULL;
}

void GaussianMixtureModel::Dump(std::string model_file)
{
	std::ofstream fout(model_file);
	dump(fout);
}

void GaussianMixtureModel::Dump(std::ofstream& fout)
{
	dump(fout);
}

void GaussianMixtureModel::Load(std::string model_file)
{
	std::ifstream fin(model_file);
	load(fin);
}

void GaussianMixtureModel::Load(std::ifstream& fin)
{
	load(fin);
}

double GaussianMixtureModel::ScoreAll(std::vector<std::vector<double>> X_in, int concurrency)
{
	//DenseDataset X;
	//conv_double_pp_to_vv(X_in, X, nr_instance, nr_dim);
	return log_probability_of_fast_exp_threaded(X_in, concurrency);
}

void GaussianMixtureModel::ScoreBatch(double ** X_in, double * prob_out, int nr_instance, int nr_dim, int concurrency)
{
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, nr_instance, nr_dim);
	std::vector<real_t> prob;
	log_probability_of_fast_exp_threaded(X, prob, concurrency);
	for (size_t i = 0; i < prob.size(); i++)
		prob_out[i] = prob[i];
}

double GaussianMixtureModel::ScoreInstance(std::vector<double>& x)
{
	//vector<real_t> x;
	//conv_double_p_to_v(x_in, x, nr_dim);
	return log_probability_of_fast_exp(x);
}

void GaussianMixtureModel::PrintParam(GMMParameter * param)
{
	printf("-->Prameters...\n");
	printf("------------------------\n");
	printf("--->nr_instance   :   %d\n", param->nr_instance);
	printf("--->nr_dim        :   %d\n", param->nr_dim);
	printf("--->nr_mixture    :   %d\n", param->nr_mixture);
	printf("--->min_covar     :   %f\n", param->min_covar);
	printf("--->threshold     :   %f\n", param->threshold);
	printf("--->nr_iteration  :   %d\n", param->nr_iteration);
	printf("--->init_with_kmeans: %d\n", param->init_with_kmeans);
	printf("--->concurrency   :   %d\n", param->concurrency);
	printf("--->verbosity     :   %d\n", param->verbosity);
	printf("------------------------\n");
}

void GaussianMixtureModel::Print_X(double ** X)
{
	printf("X: %p\n", X);
	printf("X: %p\n", X[0]);
	printf("X: %f\n", X[0][0]);
}

void GaussianMixtureModel::conv_double_pp_to_vv(double **Xp, DenseDataset &X, int nr_instance, int nr_dim) {
	X.resize(nr_instance);
	for (auto &x : X)
		x.resize(nr_dim);
	for (int i = 0; i < nr_instance; i++)
		for (int j = 0; j < nr_dim; j++)
			X[i][j] = Xp[i][j];
}

void GaussianMixtureModel::conv_double_p_to_v(double *x_in, std::vector<real_t> &x, int nr_dim) {
	x.resize(nr_dim);
	for (int i = 0; i < nr_dim; i++)
		x[i] = x_in[i];
}

