#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>
#include "VectorOperations.hpp"

namespace MachineLearning
{
	/**
	* Class of Linear Regression, designed to fit the fuction y = a * x + b
	* @param x -- input data
	* @param y -- output data
	* @param predicted_y -- vector to store predicted outputs
	* @param alpha -- a
	* @param bias -- b
	*/
	template<class T>
	class LinearRegression
	{
		/**
		* Friend function overloading the operator "<<" to print information of the model
		*/
		friend std::ostream& operator<<(std::ostream& cout, LinearRegression const& LR_model)
		{
			cout.precision(6);
			cout << "The fit function : y = " << LR_model.alpha << " * x + " << LR_model.bias;
			return cout;
		}

	public:
		std::vector<T> x, y, predicted_y;
		double alpha, bias;

		/**
		* Constructor with no parameters
		*/
		LinearRegression();
		
		/**
		* Constructor with data vectors
		*/
		LinearRegression(std::vector<T>& _x, std::vector<T>& _y); 

		/**
		* Function to start training
		* @param epoches -- count of training epoches
		* @param lr -- learning rate
		*/
		void Train(const int& epoches, const double& lr);

		/**
		* Function to forward propagate input data to get the prediction
		*/
		std::vector<T> Predict(std::vector<T> const& x);
		
		
		

	private:
		/**
		* Function to adjust the coefficient using gradient descent
		* @param lr -- learning rate
		*/
		inline void GradientDescent(double lr);

		/**
		* Function to calculate the MSE Loss of the two vectors.
		* Use identifier const to make sure we don't change the data carelessly
		*/
		inline T MSELoss(const std::vector<T>& y, const std::vector<T>& predicted_y);

		/**
		* Function to normalise the data,which is necessary for training
		*/
		

	};




	//-----------------------------------------------------------------------------------------------------
	//-------------------------Implementation of member functions------------------------------------------
	//-----------------------------------------------------------------------------------------------------
	template<class T>
	LinearRegression<T>::LinearRegression()
	{
		std::vector<double> _x, _y;
		this->x = _x;
		this->y = _y;
		this->alpha = double(rand() % 100) / 10;
		this->bias = double(rand() % 100) / 10;
	}

	template<class T>
	LinearRegression<T>::LinearRegression(std::vector<T>& _x, std::vector<T>& _y)
	{
		this->x = _x;
		this->y = _y;
		//initialize alpha, bias belongs to [0, 10]
		this->alpha = double(rand() % 100) / 10;
		this->bias = double(rand() % 100) / 10;
		std::cout << "Initialize : " << *this << std::endl << std::endl;
	}


	/**
	* Function to be used in MSELoss, which could get the square of the input
	*/
	template<class T>
	T Square(const T& x)
	{
		return x * x;
	}

	template<class T>
	T LinearRegression<T>::MSELoss(const std::vector<T>& y, const std::vector<T>& predicted_y)
	{
		if (y.size() != predicted_y.size())
		{
			std::cerr << "ERROR (" << __func__ << ") : ";
			std::cerr << "Supplied vectors have different shapes, ";
			std::cerr << "Got y : " << y.size() << " and predicted_t : " << predicted_y.size() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::vector<T> tem = predicted_y - y;
		tem = VectorApplyFunction(tem, Square);
		T mseloss = VectorSum(tem) / (2 * y.size());

		return mseloss;
	}

	template<class T>
	std::vector<T> LinearRegression<T>::Predict(std::vector<T> const& x)
	{
		std::vector<T> predicted_output;
		for (auto i : x)
		{
			predicted_output.emplace_back(alpha * i + bias);
		}
		return predicted_output;
	}

	template<class T>
	void LinearRegression<T>::GradientDescent(double lr)
	{
		double GradientSum_a = 0;
		double GradientSum_b = 0;

		//We shouldn't change values of a and b before calculating their average gradient
		//alpha : The derivative of MSELoss to a is { (ax + b - y) * x => (predicted_y - y) * x }
		for (int i = 0; i < x.size(); i++)
		{
			GradientSum_a += (predicted_y[i] - y[i]) * x[i];
		}

		//bias : The derivative of MSELoss to b is { (ax + b - y) * 1 => (predicted_y - y) }
		for (int i = 0; i < x.size(); i++)
		{
			GradientSum_b += (predicted_y[i] - y[i]);
		}

		std::cout << "Gradient_a = " << GradientSum_a / x.size() << std::endl;
		std::cout << "Gradient_b = " << GradientSum_b / x.size() << std::endl;
		alpha -= lr * GradientSum_a / x.size();
		bias -= lr * GradientSum_b / x.size();

	}

	template<class T>
	void LinearRegression<T>::Train(const int& epoches, const double& lr)
	{

		for (int epoch = 0; epoch < epoches; epoch++)
		{
			predicted_y = Predict(x);
			double mseloss = MSELoss(y, predicted_y);
			GradientDescent(lr);
			std::cout << "epoch = " << epoch << ", MSELoss = " << mseloss << std::endl;
			std::cout << *this << std::endl << std::endl;

		}
	}


}



#endif