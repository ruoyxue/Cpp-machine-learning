#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>
#include <valarray>
#include "VectorOperations.hpp"

namespace MachineLearning
{
	class Perceptron
	{
	public:
		//@param feature_dim -- dimension of input x
		Perceptron(int feature_dim);

		//@brief Function to calculate the sum of loss of the mislabeled data 
		double SumLoss(std::vector<std::pair<std::valarray<double>, double>>& dataset);

		void Train(std::vector<std::pair<std::valarray<double>, double>>& dataset, double lr, bool shuffle);

		//@brief Function to get prediction from single data x
		inline double Predict_SingleX(std::valarray<double>& x);

		static void Visualization();
	private:
		std::valarray<double> w;
		double b;

	};


	//-----------------------------------------------------------------------------------------------------
	//-------------------------Implementation of member functions------------------------------------------
	//-----------------------------------------------------------------------------------------------------
	Perceptron::Perceptron(int feature_dim)
	{
		this->w.resize(feature_dim);
		for (double& i : w) i = 0;
		b = 0;
	}

	double Perceptron::SumLoss(std::vector<std::pair<std::valarray<double>, double>>& dataset)
	{
		double loss = 0, predicted_y = 0;
		for (auto data : dataset)
		{
			predicted_y = Predict_SingleX(data.first);
			if (predicted_y != data.second)
				loss += -data.second * ((this->w * data.first).sum() + this->b);
		}
		return loss;
	}

	inline double Perceptron::Predict_SingleX(std::valarray<double>& x)
	{
		return ((this->w * x).sum() + this->b) >= 0 ? 1 : -1;
	}

	void Perceptron::Train(std::vector<std::pair<std::valarray<double>, double>>& dataset, double lr = 0.5, bool shuffle = false)
	{
		if (lr <= 0 || lr > 1) lr = 0.5;
		int epoch = 0;
		double sum_loss = this->SumLoss(dataset);
		do{
			epoch++;
			for (auto const data : dataset)
			{
				while((data.second * ((this->w * data.first).sum() + this->b)) <= 0)
				{
					this->w += lr * data.first * data.second;
					this->b += lr * data.second;
				}
			}
			sum_loss = this->SumLoss(dataset);
			std::cout << "epoch = " << epoch << " , sum_loss = " << sum_loss << std::endl;
			if (shuffle) VectorShuffle(dataset);

		} while (this->SumLoss(dataset) != 0);

	}


















}




#endif 
