#ifndef VECTOR_OPERATIONS_HPP
#define VECTOR_OPERATIONS_HPP
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

namespace MachineLearning
{
	/**
	*/
	template<typename T>
	std::ostream& operator<<(std::ostream& cout, const std::vector<T>& A)
	{
		for (auto i : A)
		{
			cout << i << " ";
		}
		return cout;
	}


	/**
	* Overloaded operator "+" to add two 2D vectors
	* @param T -- typename of the vector
	* @param A -- First 2D vector
	* @param B -- Second 2D vector
	* @return new resultant vector
	*/
	template <typename T>
	std::vector<T> operator+(const std::vector<T>& A, const std::vector<T>& B)
	{
		//If vectors don't have equal shape
		if (A.size() != B.size())
		{
			std::cerr << "ERROR (" << __func__ << ") : ";
			std::cerr << "Supplied vectors have different shapes, ";
			std::cerr << "Got A : " << A.size() << " and B : " << B.size() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::vector<T> output;
		for (size_t i = 0; i < A.size(); i++)
		{
			output.emplace_back(A[i] + B[i]);
		}

		return output;
	}

	/**
	* Overloaded operator "-" to subtract one 2D vectors from the other
	* @tparam T -- typename of the vector
	* @param A -- First 2D vector
	* @param B -- Second 2D vector
	* @return new resultant vector
	*/
	template <typename T>
	std::vector<T> operator-(const std::vector<T>& A, const std::vector<T>& B)
	{
		//If vectors don't have equal shape
		if (A.size() != B.size())
		{
			std::cerr << "ERROR (" << __func__ << ") : ";
			std::cerr << "Supplied vectors have different shapes, ";
			std::cerr << "Got A : " << A.size() << " and B : " << B.size() << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::vector<T> output;
		for (size_t i = 0; i < A.size(); i++)
		{
			output.emplace_back(A[i] - B[i]);
		}
		return output;
		
	}

	/**
	* Function to add up the elements of the vector
	* @param T -- typename of the vector
	* @param A -- 2D vector
	* @return sum of elements in A
	*/
	template<typename T>
	T VectorSum(const std::vector<T>& A)
	{
		T sum = 0;
		for (size_t i = 0; i < A.size(); i++)
		{
			sum += A[i];
		}
		return sum;
	}

	/**
	* Function to normalize the vector, using Z-normalization
	* @param T -- typename of the vector
	* @param A -- 2D vector
	*/
	template<typename T>
	void VectorZNorm(std::vector<T>& A)
	{
		
		double mean = 0, std = 0, tem = 0;
		for (auto i : A)
		{
			tem += i;
		}
		mean = tem / A.size();
		tem = 0;
		for (auto i : A)
		{
			tem += (i - mean) * (i - mean);
		}
		std = sqrt(tem / A.size());
		for (auto& i : A)
		{
			i = (i - mean) / std;
		}
		
	}

	/**
	* Function to apply function to each element of the vector
	* @param T -- typename of the vector
	* @param A -- 2D vector
	* @return new resultant vector
	*/
	template<typename T>
	std::vector<T> VectorApplyFunction(const std::vector<T>& A, T(*func)(const T&))
	{
		std::vector<T> B = A;
		for (auto &b : B)
		{
			b = func(b);
		}
		return B;
	}

}


































#endif