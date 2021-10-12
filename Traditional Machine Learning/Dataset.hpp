#ifndef DATASET_HPP
#define DATASET_HPP
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>
#include <valarray>

#define RandomDouble(a, b) double(rand()%1001) / 1000 *((b) - (a)) + (a)

namespace MachineLearning
{
	/**
	* @brief Base class for different datasets
	* @param count -- the size of dataset
	* @param feature_dim -- feature dimensions
	* @param max -- maximum of x
	* @param min -- minimum of x
	* @param train -- the prepared dataset for training
	* @param test -- the prepared dataset for testing
	*/
	class Dataset
	{
	public:
		Dataset() :count(20), feature_dim(2), min(-10.), max(10.) {}
		Dataset(int count) : count(count), feature_dim(2), min(-10.), max(10.) {}
		Dataset(int count, int feature_dim) : count(count), feature_dim(feature_dim), min(-10.), max(10.) {}
		Dataset(int count, int feature_dim, double min, double max) : count(count), feature_dim(feature_dim), min(min), max(max) {}

		//@brief Function to output prepared dataset for training
		std::vector<std::pair<std::valarray<double>, double>> GetTrainDataset()
		{
			for (int i = 0; i < this->count * 4 / 5; i++)
				this->train.emplace_back(RandomPair());
			return train;
		}
		
		//@brief Function to output prepared dataset for training
		std::vector<std::pair<std::valarray<double>, double>> GetTestDataset()
		{
			for (int i = 0; i < this->count / 5; i++)
				this->test.emplace_back(RandomPair());
			return test;
		}


	protected:
		int count, feature_dim;
		double max, min;
		std::vector<std::pair<std::valarray<double>, double>> train, test;
		
		//@brief Function to get pairs of x and y, namely single data
		virtual inline std::pair<std::valarray<double>, double> RandomPair() = 0;		

	};

	/**
	* @brief Create linearly separable classification dataset for perceptron.
	* @param w, b -- separating hyperplane would be (w¡¤x + b = 0)
	*/
	class PerceptronDataset : public Dataset
	{
	public:
		PerceptronDataset() :Dataset() { GetHyperplane(); }
		PerceptronDataset(int count) :Dataset(count) { GetHyperplane(); }
		PerceptronDataset(int count, int feature_dim):Dataset(count, feature_dim){ GetHyperplane(); }
		PerceptronDataset(int count, int feature_dim, double min, double max):Dataset(count, feature_dim, min, max){ GetHyperplane(); }

	private:
		std::valarray<double> w;
		double b;
		
		//@brief Function to get the parameters of separating hyperplane
		void GetHyperplane()
		{
			w.resize(this->feature_dim);
			for (double& i : w)
				i = RandomDouble(-10., 10.);
			b = RandomDouble(-10. ,10.);
		}

		inline std::pair<std::valarray<double>, double> RandomPair()
		{
			std::pair<std::valarray<double>, double> pair;
			std::valarray<double> x(this->feature_dim);

			//We would not like to have points on the separating hyperplane
			do
			{
				for (double& i : x)
					i = RandomDouble(this->min, this->max);
			} while (((this->w * x).sum() + b) == 0);

			double y = (((this->w * x).sum() + b) > 0) ? 1 : -1;
			return std::make_pair(x, y);
		}

	};

	/**
	* @brief Create linearly separable classification dataset for perceptron.
	* @param w, b -- the fitting target would be (w¡¤x + b = 0)
	*/
	class LinearRegressionDataset : public Dataset
	{
	public:
		LinearRegressionDataset() :Dataset() { GetTarget(); }
		LinearRegressionDataset(int count) :Dataset(count) { GetTarget(); }
		LinearRegressionDataset(int count, int feature_dim) :Dataset(count, feature_dim) { GetTarget(); }
		LinearRegressionDataset(int count, int feature_dim, double min, double max) :Dataset(count, feature_dim, min, max) { GetTarget(); }
		std::valarray<double> w;
		double b;

	private:
		std::default_random_engine generator;
		std::normal_distribution<> norm{ 0, 0.05 };// normal distribution

		//@brief Function to get the parameters of the fitting target
		void GetTarget()
		{
			w.resize(this->feature_dim);
			for (double& i : w)
				i = RandomDouble(-10., 10.);
			b = RandomDouble(-10., 10.);
		}

		inline std::pair<std::valarray<double>, double> RandomPair()
		{
			std::pair<std::valarray<double>, double> pair;
			std::valarray<double> x(this->feature_dim);

			for (double& i : x) 
				i = RandomDouble(this->min, this->max);

			double y = (this->w * x).sum() + b + norm(generator);
			return std::make_pair(x, y);
		}

	};

}



















#endif