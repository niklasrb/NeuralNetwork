#ifndef NNNETWORK_H
#define NNNETWORK_H


#include "Layer.h"
#include <string>

namespace NeuralNetwork
{


class Network
{
friend class Trainer;
protected:
	std::vector<Layer> layers;
public:

	void Append(const Layer& L);
	void Append(int outputLen);
	void Append(int outputLen, std::shared_ptr<ActivationFunction> af);
	
		
	int InputLen() const;
	int OutputLen() const;
	int numLayers() const;
	Layer& Layers(int i);
	
	void Save(std::ostream& os) const;
	void Load(std::istream& is, const std::map<std::string, std::shared_ptr<ActivationFunction> >& mapToFunc);
	void Load(std::istream& is);
	
	Vector Propagate(const Vector& input) const;
	
	void Dump() const;
	void Info() const;
	
	friend Network& operator +=(Network& N1, const Network& N2);

	friend Network operator *=(Network& N, const Numerical& a);
	
	Layer& GetLayer(int i);
	
};


}

#endif
