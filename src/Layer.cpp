#include "../nnetwork.h"

namespace NeuralNetwork 
{

void Layer::Propagate(Vector& v) const
{
	assert(false); // something is wrong with this function
	assert(v.size() == inputLen);
	v = W*v + b;
	for(int i = 0; i < outputLen; i++)
		v[i] = (*s)(v[i]);
}

Vector Layer::Propagate(const Vector& v) const
{
	assert(v.size() == inputLen);
	Vector w = W*v + b;
	for(int i = 0; i < outputLen; i++)
		w[i] = (*s)(w[i]);
	return w;
}

Layer::Layer() : Layer(0, 0)
 {}

Layer::Layer(int input, int output) : Layer(input, output, std::make_shared<RELU>())
 {}

Layer::Layer(int input, int output, std::shared_ptr<ActivationFunction> s) : inputLen(input) , outputLen(output), W(output, input), b(output), s(s)
 {}
 
void Layer::Save(std::ostream& os) const
{
	assert(os.good());
	assert(inputLen > 0 && outputLen > 0);
	os << inputLen << "\t" << outputLen << std::endl;
	os << W  << std::endl << b << std::endl;
	os << s->Name() << std::endl;
}

Matrix readMatrix(std::istream& is, int m, int n) 
{
	Matrix M(m, n);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			assert(is.good());
			is >> M(i,j);
		}
	}
	return M;
}

Vector readVector(std::istream& is, int n)
{
	Vector v(n);
	for(int i =0; i < n; i++) {
		assert(is.good());
		is >> v(i);
	}
	return v;
}

void Layer::InitializeRandomly() 
{
	W.setRandom(outputLen, inputLen);
	b.setRandom(outputLen);
}

void Layer::InitializeZero() 
{
	W = Matrix::Zero(outputLen, inputLen);
	b = Vector::Zero(outputLen);
}

void Layer::Load(std::istream& is, const std::map<std::string, std::shared_ptr<ActivationFunction> >& mapToFunc)
{
	assert(is.good());
	is >> inputLen; is >> outputLen;
	assert(inputLen > 0 && outputLen > 0);
	W = readMatrix(is, outputLen, inputLen);
	b = readVector(is, outputLen);
	assert(is.good());
	std::string n;
	is >> n;
	assert(mapToFunc.find(n) != mapToFunc.end());
	s = mapToFunc.at(n);
}

void Layer::Dump() const
{
	std::cout << "W = " << W << std::endl << " b = " << b << std::endl << s->Name() << std::endl;
}

Layer& operator +=(Layer& L, const Layer& delL)
{
	L.W += delL.W;
	L.b += delL.b;
	return L;
}

Layer& operator *= (Layer& L, const Numerical& x )
{
	L.W *= x;
	L.b *= x;
	return L;
}

Numerical Layer::L1() const
{
	Numerical s = 0;
	for(int i = 0; i < W.rows(); i++)
		for(int j = 0; j < W.cols(); j++)
			s += std::abs(W(i,j));
	for(int i = 0; i < b.size(); i++)
		s += std::abs(b[i]);
	return s;
}

Numerical Layer::L2() const
{
	Numerical s = 0;
	for(int i = 0; i < W.rows(); i++)
		for(int j = 0; j < W.cols(); j++)
			s += std::pow(W(i,j), 2);
	for(int i = 0; i < b.size(); i++)
		s += std::pow(b[i], 2);
	return s;
}

}
