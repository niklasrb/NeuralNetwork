#include "../nnetwork.h"

namespace NeuralNetwork 
{


Numerical RELU::operator() (const Numerical& x) const
{
	return (x > 0. ? x : 0.);
}
	
Numerical RELU::Derivative(const Numerical & x) const
{
	return (x > 0. ? 1. : 0.);
}
	
std::string RELU::Name() const
{
	return std::string("RELU");
}

Numerical Logistic::operator() (const Numerical& x) const
{
	return 1./(1.+exp(-x));
}
	
Numerical Logistic::Derivative(const Numerical & x) const
{
	return exp(x)/pow(1.+exp(x), 2);
}
	
std::string Logistic::Name() const
{
	return std::string("Logistic");
}

}
