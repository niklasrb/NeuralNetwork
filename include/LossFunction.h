#ifndef NNLOSSFUNCTION_H
#define NNLOSSFUNCTION_H

#include "Const.h"
#include <string>
#include <cmath>

namespace NeuralNetwork 
{
	
class LossFunction
{
public:
	Numerical virtual operator()(Vector output, Vector expectation) const = 0;
	Vector virtual Derivative(Vector output, Vector expectation) const = 0;
};

class EuclideanDistance : public LossFunction
{
public:
	Numerical operator()(Vector output, Vector expectation) const;
	Vector Derivative(Vector output, Vector expectation) const;
};

}

#endif


