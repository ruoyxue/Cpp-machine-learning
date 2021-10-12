#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>
#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Perceptron.hpp"

using namespace MachineLearning;

//@brief test linear regression
void TestLinearRegression()
{
	int feature_dim = 7, dataset_size = 1500;
	double max = 10, min = -10;
	double lr = 0.01, epoches = 500;

	LinearRegressionDataset LR_dataset(dataset_size, feature_dim, min, max);
	std::vector<std::pair<std::valarray<double>, double>>
		train_dataset = LR_dataset.GetTrainDataset(), test_dataset = LR_dataset.GetTestDataset();
	
	LinearRegression LR_model(feature_dim);
	LR_model.Train(train_dataset, epoches, lr, true);

	std::cout << "Target Parameters : " << std::endl << " w : ";
	for (int i = 0; i < LR_dataset.w.size(); i++)
		std::cout << LR_dataset.w[i] << "   ";
	std::cout << "   b : " << LR_dataset.b << std::endl;
}

//@brief test perceptron
void TestPerceptron()
{
	int feature_dim = 5, dataset_size = 1500;
	double max = 100, min = -100;
	double lr = 0.5;

	PerceptronDataset P_dataset(dataset_size, feature_dim, min, max);
	std::vector<std::pair<std::valarray<double>, double>> 
		train_dataset = P_dataset.GetTrainDataset(), test_dataset = P_dataset.GetTestDataset();
	Perceptron P_model(feature_dim);
	P_model.Train(train_dataset, lr);
	std::cout << "test sum loss = " << P_model.SumLoss(test_dataset) << std::endl;

}

int main()
{
	srand((int)std::time(0));
	//TestPerceptron();
	TestLinearRegression();
	return 0;
}