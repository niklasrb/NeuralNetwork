#ifndef NNLAYER_H
#define NNLAYER_H


#include "Const.h"
#include "ActivationFunction.h"
#include <iostream>
#include <map>
#include <memory>

namespace NeuralNetwork 
{
class Trainer;
class Network;
	
	
class Layer
{
friend class Network;
friend class Trainer;
protected:
	int inputLen, outputLen;
	Matrix W;
	Vector b;
	std::shared_ptr<ActivationFunction> s;
	
public:
	Layer();
	Layer(int input, int output);
	Layer(int input, int output,  std::shared_ptr<ActivationFunction> s);
	
	void InitializeRandomly();
	void InitializeZero();
	void Save(std::ostream& o) const;
	void Load(std::istream& i, const std::map<std::string, std::shared_ptr<ActivationFunction> >& mapToFunc);
	
	Vector Propagate(const Vector& v) const;
	void Propagate(Vector& v) const;
	
	void Backpropagate();
	
	void Dump() const;
	
	friend Layer& operator +=(Layer& L, const Layer& delL);
	friend Layer& operator *=(Layer& L, const Numerical& x); 
	
	Numerical L1() const;
	Numerical L2() const;
};

}

#endif
