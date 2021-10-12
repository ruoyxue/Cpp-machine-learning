#ifndef VECTOR_OPERATIONS_HPP
#define VECTOR_OPERATIONS_HPP
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <valarray>

namespace MachineLearning
{

	void VectorShuffle(std::vector<std::pair<std::valarray<double>, double>>& dataset)
	{
		int len = dataset.size();
		int a = 0, b = 0;
		std::pair<std::valarray<double>, double> tem;
		int count = len / 2;

		while (count--)
		{
			if (a != b)
			{
				tem = dataset[a];
				dataset[a] = dataset[b];
				dataset[b] = tem;
			}

			a = rand() % len;
			b = rand() % len;

		}

	}

	
}


































#endif