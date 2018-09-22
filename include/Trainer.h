#ifndef NNTRAINER_H
#define NNTRAINER_H


#include "Network.h"
#include "LossFunction.h"
#include <random>
#include <algorithm>

namespace NeuralNetwork
{

typedef std::pair<Vector, Vector> DataPoint;


class Trainer
{
public:
	enum RegulatoryOptions {
			none, L1, L2
		};

	struct TrainingOptions
	{
		Numerical trainingRate;
		Numerical trainingRateMin;
		bool adjustTrainingRate;
		bool useInertia;
		Numerical inertiaCoefficient;
		int maxBatches;
		int maxFails;
		int batchSize;
		Numerical objectiveLoss;
		RegulatoryOptions regulationType;
		Numerical regulationCoefficient;
		
		
		TrainingOptions() : trainingRate(0.03), trainingRateMin(0.0001), adjustTrainingRate(true), useInertia(false), inertiaCoefficient(0.2), maxBatches(100), maxFails(50),
				batchSize(32), objectiveLoss(0.), regulationType(RegulatoryOptions::none), regulationCoefficient(0.3) { }
	};
protected:
	std::shared_ptr<Network> NN;
	std::shared_ptr<LossFunction> Loss;	
	
	std::shared_ptr<Network> Batch(const std::vector<DataPoint>& batch) const;
	void Backpropagation(const DataPoint& dp, std::shared_ptr<Network> NNcp) const;
	void RegulationDifferentiation(std::shared_ptr<Network> nn, std::shared_ptr<Network> nnUpdate, RegulatoryOptions ro, Numerical regulationCoefficient) const;
public:
	
	TrainingOptions Options;
	std::vector<DataPoint> testSet;
	std::vector<DataPoint> trainSet;
	
	int Training();
	Numerical Eval(const std::vector<DataPoint>& batch) const;
	Numerical Eval() const;
	
	Numerical RegulationPenalty(RegulatoryOptions ro) const;
	
	Trainer(std::shared_ptr<Network> NN, std::shared_ptr<LossFunction> Loss);
	Trainer(std::shared_ptr<Network> NN, std::shared_ptr<LossFunction> Loss, TrainingOptions to);
};


}

#endif
