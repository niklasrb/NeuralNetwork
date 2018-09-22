#ifndef NNACTIVATIONFUNCTION_H
#define NNACTIVATIONFUNCTION_H

#include "Const.h"
#include <string>
#include <cmath>

namespace NeuralNetwork 
{
	
	
class ActivationFunction
{
public:
	Numerical virtual operator ()(const Numerical& x) const = 0;
	Numerical virtual Derivative(const Numerical& x) const = 0;	
	std::string virtual Name() const = 0;
	
};

class RELU : public ActivationFunction
{
public:
	Numerical operator() (const Numerical& x) const;
	Numerical Derivative(const Numerical & x) const;
	std::string Name() const;
};

class Logistic : public ActivationFunction
{
public:
	Numerical operator() (const Numerical& x) const;
	Numerical Derivative(const Numerical & x) const;
	std::string Name() const;
};

}

#endif

