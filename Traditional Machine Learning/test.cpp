#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>
#include "LinearRegression.hpp"


/**
* The function to create data for Linear Regression.
* We expect to fit the function y = 3x + 4 (single variable for simplicity), 
* use std::vector to store the data to be trained, which is more flexible than the fixed-size array.
* @param x -- input data
* @param y -- output data
* @param PrintData -- whether to print x, y or not
*/
void CreateDataForLR(std::vector<double>& x, std::vector<double>& y, bool PrintData = false)
{
	//generating random data
	srand((int)std::time(0));
	std::default_random_engine generator;
	std::normal_distribution<> norm{ 0, 0.1 };//mu:0 sigma:0.1, normal distribution
	for (double i = 0; i < 100; i++)
	{
		double random = double(rand() % 100) / 25;
		x.emplace_back(random);
		y.emplace_back(3 * random + 4 + norm(generator));
		
	}

	if (PrintData)
	{
		std::cout << "Training data:" << std::endl << "x : ";
		//two methods to traverse vectors
		for (std::vector<double>::const_iterator i = x.begin(); i != x.end(); i++)
		{
			std::cout << *i << ' ';
		}
		std::cout << std::endl << "\n" << "y : ";
		for (double i : y)
		{
			std::cout << i << " ";
		}
		std::cout << std::endl;
	}
	
	return;

}


/**
* The function to test my Linear Regression model.
*/
void test()
{
	std::vector<double> x, y;
	CreateDataForLR(x, y, false);
	MachineLearning::LinearRegression<double> LR_model(x, y);
	LR_model.Train(500, 0.1);

}

int main()
{
	test();

	return 0;
}