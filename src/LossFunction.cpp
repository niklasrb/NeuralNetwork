
#include "../nnetwork.h"

namespace NeuralNetwork
{

Numerical EuclideanDistance::operator ()(Vector output, Vector expectation) const
{
	return 1./2. *pow((output - expectation).norm(), 2);
}

Vector EuclideanDistance::Derivative(Vector output, Vector expectation) const
{
	return output - expectation;
}

}




