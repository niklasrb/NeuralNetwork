#include "../nnetwork.h"

namespace NeuralNetwork
{


void Network::Append(const Layer& L)
{
	if(layers.size() > 0)
		assert(layers.at(layers.size()-1).outputLen == L.inputLen);
	layers.push_back(L);
}

void Network::Append(int outputLen) 
{
	Append(outputLen, std::make_shared<RELU>());
}

void Network::Append(int outputLen, std::shared_ptr<ActivationFunction> af)
{
	assert(layers.size() > 0);
	Append(Layer(layers.at(layers.size()-1).outputLen, outputLen, af));
}

int Network::InputLen() const
{
	if(layers.size() == 0)
		return 0;
	else 
		return layers.at(0).inputLen;
}

int Network::OutputLen() const
{
	if(layers.size() == 0)
		return 0;
	else
		return layers.at(layers.size()-1).outputLen;
}

int Network::numLayers() const
{
	return layers.size();
}

Layer& Network::Layers(int i)
{
	return layers.at(i);
}

void Network::Save(std::ostream& os) const
{
	assert(os.good());
	assert(layers.size() > 0);
	os << layers.size() << std::endl;
	for(auto it = layers.begin(); it != layers.end(); ++it)
		it->Save(os);
}

void Network::Load(std::istream& is, const std::map<std::string, std::shared_ptr<ActivationFunction> >& mapToFunc)
{
	assert(is.good());
	layers.clear();
	int n; is >> n;
	assert(n > 0);
	for(int i = 0; i < n; i++) {
		Layer L;
		L.Load(is, mapToFunc);
		layers.push_back(L);
	}
}

void Network::Load(std::istream& is)
{
	 std::map<std::string, std::shared_ptr<ActivationFunction> > mapToFunc;
	 mapToFunc["RELU"] = std::make_shared<RELU>();
	 mapToFunc["Logistic"] = std::make_shared<Logistic>();
	 
	 Load(is, mapToFunc);
}

Vector Network::Propagate(const Vector& input) const
{
	std::vector<Vector> res; res.reserve(layers.size()+1);
	res.push_back(input);
	for(uint i = 0; i < layers.size(); i++) {
		//std::cout << "Layer " << i << std::endl;
		res.push_back(layers.at(i).Propagate((const Vector&)res.at(i)));
		//std::cout << "v_" << i+1 << " = " << res.at(i+1) << std::endl;
	}
	return res.at(res.size()-1);
}

void Network::Dump() const
{
	std::cout << "Neural Network with " << numLayers() << "Layers" << std::endl;
	for(uint i = 0 ; i < layers.size(); i++) {
		std::cout << i << ": " << layers.at(i).inputLen << " -> " << layers.at(i).outputLen << std::endl;
		layers.at(i).Dump();
	}
}

void Network::Info() const
{
	std::cout << "Neural Network with " << numLayers() << "Layers" << std::endl;
	for(uint i = 0 ; i < layers.size(); i++) {
		std::cout << i << ": " << layers.at(i).inputLen << " -> " << layers.at(i).outputLen << std::endl;
	}
}

Network& operator +=(Network& N1, const Network& N2)
{
	assert(N1.numLayers() == N2.numLayers());
	for(int i = 0; i < N1.numLayers(); i++)
		N1.layers.at(i) += N2.layers.at(i);
	return N1;
}

Network operator *=(Network& N, const Numerical& a)
{
	for(int i = 0; i < N.numLayers(); i++)
		N.layers.at(i) *= a;
	return N;
}

Layer& Network::GetLayer(int i)
{
	return layers.at(i);
}

}
