#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>
#include "VectorOperations.hpp"

#define RandomDouble(a, b) double(rand()%1001) / 1000 *((b) - (a)) + (a)

namespace MachineLearning
{
	class LinearRegression
	{
	public:
		//@param feature_dim -- dimension of input x
		LinearRegression(int feature_dim);

		void Train(std::vector<std::pair<std::valarray<double>, double>>& dataset,
			const int& epoches, const double& lr, bool shuffle);

		//@brief Function to get regression output from single data x
		inline double Predict_SingleX(std::valarray<double>& x);
	
	private:
		std::valarray<double> w;
		double b;

		inline void GradientDescent(std::vector<std::pair<std::valarray<double>, double>>& dataset, double lr);

		inline double Loss(double y, double predicted_y);


	};

	//-----------------------------------------------------------------------------------------------------
	//-------------------------Implementation of member functions------------------------------------------
	//-----------------------------------------------------------------------------------------------------
	LinearRegression::LinearRegression(int feature_dim)
	{
		this->w.resize(feature_dim);

		//We can't initialise w = 0
		for (double& i : w)
		{
			do 
			{
				i = RandomDouble(-10., 10.);
			} while (i == 0);

		}
		b = RandomDouble(-10., 10.);
	}

	double LinearRegression::Predict_SingleX(std::valarray<double>& x)
	{
		return (this->w * x).sum() + this->b;
	}

	double LinearRegression::Loss(double predicted_y, double y)
	{
		return (predicted_y - y) * (predicted_y - y);
	}

	void LinearRegression::Train(std::vector<std::pair<std::valarray<double>, double>>& dataset,
												const int& epoches, const double& lr = 0.01, bool shuffle= false)
	{
		double predicted_y = 0, loss = 0;
		for (int epoch = 0; epoch < epoches; epoch++)
		{
			for (auto data : dataset)
			{
				predicted_y = Predict_SingleX(data.first);
				loss += Loss(predicted_y, data.second);
			}
			loss /= dataset.size();
			GradientDescent(dataset, lr);

			std::cout << "epoch = " << epoch << ", Loss = " << loss << std::endl;
			std::cout << "Parameters " << " w : ";
			for (int i = 0; i < this->w.size(); i++)
				std::cout << this->w[i] << "   ";
			std::cout << "   b : " << this->b << std::endl;

			if (shuffle)
				VectorShuffle(dataset);
		}
	}


	void LinearRegression::GradientDescent(std::vector<std::pair<std::valarray<double>, double>>& dataset,double lr)
	{
		std::valarray<double> GradientSum_w(this->w.size());
		double GradientSum_b = 0;
		double predicted_y = 0;
		for (double& i : GradientSum_w) i = 0;

		for (auto data : dataset)
		{
			predicted_y = Predict_SingleX(data.first);
			for (int i = 0; i < this->w.size(); i++)
				GradientSum_w[i] += (predicted_y - data.second) * data.first[i];
			GradientSum_b += predicted_y - data.second;
		}
		
		for (int i = 0; i < this->w.size(); i++)
			this->w[i] -= lr * GradientSum_w[i] / dataset.size();
		b -= lr * GradientSum_b / dataset.size();

	}

	


}



#endif