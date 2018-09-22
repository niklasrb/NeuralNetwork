
#include "../nnetwork.h"

namespace NeuralNetwork 
{

void Trainer::Backpropagation(const DataPoint& dp, std::shared_ptr<Network> NNcp) const
{
	std::vector<Vector> outputs; outputs.reserve(this->NN->numLayers()+1);
	outputs.push_back(dp.first);		// contains output of layer i at i+1
	for(int i = 1; i < this->NN->numLayers()+1; i++) {
		outputs.push_back(this->NN->layers.at(i-1).Propagate((const Vector&)outputs.at(i-1)));
	}
	std::vector<Vector> EdiffOutput; EdiffOutput.reserve(this->NN->numLayers());	// contains del E/ del o_j in descending order
	EdiffOutput.push_back(Loss->Derivative(outputs.at(outputs.size()-1), dp.second));
	
	// go from the last layer to the first layer and compute all necessary intermediate values
	for(int z = this->NN->numLayers()-1; z >= 0; z--) {	// don't make this uint!
		//std::cout << "Layer: " << z << std::endl;
		Layer &L = NN->layers.at(z);
		Layer &Lcp = NNcp->layers.at(z);
		Vector Ediff = Vector::Zero(L.inputLen);
		for(int i = 0; i < L.outputLen; i++) {
			for(int j = 0; j < L.inputLen; j++) {
				Lcp.W(i, j) += EdiffOutput.at(this->NN->numLayers()-1 - z)(i) *
								L.s->Derivative(outputs.at(z+1)(i)) * outputs.at(z)(j);
				Ediff(j) +=	EdiffOutput.at(this->NN->numLayers()-1 - z)(i) *
								L.s->Derivative(outputs.at(z+1)(i)) * L.W(i, j);
			}
			Lcp.b(i) += EdiffOutput.at(this->NN->numLayers()-1 - z)(i) *
								L.s->Derivative(outputs.at(z+1)(i)) ;
		}
		//std::cout << "Lcp.W = " <<  Lcp.W << std::endl << "Lcp.b" << Lcp.b << std::endl;
		//std::cout << "Ediff = " << Ediff << std::endl;
		EdiffOutput.push_back(Ediff);
	}
}

Numerical Trainer::Eval(const std::vector<DataPoint>& batch) const
{
	//std::cout << "eval: " << std::endl;
	Numerical s = 0;
	for(uint i = 0 ; i < batch.size(); i++) {
		//std::cout << NN->Propagate(batch.at(i).first) << std::endl << " vs " << batch.at(i).second << std::endl ;
		s +=(* this->Loss)(NN->Propagate(batch.at(i).first), batch.at(i).second);
		//std::cout << s << std::endl;
	}
	return s;
}

Numerical Trainer::Eval() const
{
	return Eval(testSet);
}

Trainer::Trainer(std::shared_ptr<Network> NN, std::shared_ptr<LossFunction> Loss) :  NN(NN), Loss(Loss)
{ }

Trainer::Trainer(std::shared_ptr<Network> NN, std::shared_ptr<LossFunction> Loss, TrainingOptions to) : NN(NN), Loss(Loss),  Options(to)
{ }

Numerical Trainer::RegulationPenalty(RegulatoryOptions ro) const
{
	if(ro == RegulatoryOptions::none)
		return 0.;
	Numerical s = 0.;
	for(int l = 0; l < NN->numLayers(); l++) {
		if(ro == RegulatoryOptions::L1)
			s += NN->layers.at(l).L1();
		if(ro == RegulatoryOptions::L2)
			s += NN->layers.at(l).L2();
	}
	return s;
}


void Trainer::RegulationDifferentiation(std::shared_ptr<Network> nn, std::shared_ptr<Network> nnUpdate, RegulatoryOptions ro, Numerical regulationCoefficient) const
{
	if(ro == RegulatoryOptions::none)
		return;
	Numerical (*task)(const Numerical&) ;
	if(ro == RegulatoryOptions::L1)
		task = [](const Numerical& x) { return (x > 0 ? (Numerical)1. : (x< 0 ? (Numerical)-1. : (Numerical)0.)); };
	if(ro == RegulatoryOptions::L2)
		task = [](const Numerical& x) { return 2*x; };
	for(int l = 0; l < nn->numLayers(); l++) {
		for(int i = 0; i < nn->layers.at(l).outputLen; i++) {
			for(int j = 0; j < nn->layers.at(l).inputLen; j++) {
				nnUpdate->layers.at(l).W(i,j) += regulationCoefficient*task(nn->layers.at(l).W(i,j));
			}
			nnUpdate->layers.at(l).b(i) += regulationCoefficient*task(nn->layers.at(l).b(i));
		}
	}
}


int Trainer::Training()
{
	assert(trainSet.size() > 0);
	TrainingOptions to = this->Options;
	int run = 0; bool improvement = true; 
	int runImprovement = 0;	
	int batchSize = (to.batchSize > 0 ? std::min(to.batchSize, (int)trainSet.size()) :(int) trainSet.size()*0.01);
	std::shuffle(trainSet.begin(), trainSet.end(), std::default_random_engine());
	Numerical Loss = this->Eval(trainSet) + to.regulationCoefficient*this->RegulationPenalty(to.regulationType);
	std::vector<DataPoint> batch;
	std::shared_ptr<Network> nnUpdate, nnPrevUpdate;
	
	while((to.maxBatches > 0 ? run < to.maxBatches : true) 	// if maxBatches is set, only continue if run is smaller
		&& (to.objectiveLoss > 0 ?  Loss > to.objectiveLoss : true) 	// if objectiveLoss is set, only continue if Loss is bigger
			&& (to.maxFails > 0 ? runImprovement > -to.maxFails : true ))	// only continue if the algorithm hasn't failed the last maxFail/100 tries - 
	{
		std::cout << "--------------" << std::endl
			<< "Batch " << run << std::endl
			<< "Loss = " << Loss << std::endl
			<< "trainingRate = " << to.trainingRate << std::endl;
		
		int s = (run*batchSize) % trainSet.size();		// choose stochastic batch
		batch = std::vector<DataPoint>(trainSet.begin() + s, trainSet.begin() + std::min(s+batchSize, (int)trainSet.size()));
		
		std::cout << "Batch Size " << batch.size() << std::endl;
		
		nnUpdate = Batch(batch);			// compute changes to be made
		 *nnUpdate *= 1./Numerical(batch.size());
		
		RegulationDifferentiation(NN, nnUpdate, to.regulationType, to.regulationCoefficient);	// does regulation require additional changes?
		
		if(to.useInertia && nnPrevUpdate) {					// inertia?
			assert(to.inertiaCoefficient >= 0. && to.inertiaCoefficient <= 1.);
			*nnUpdate *= -to.trainingRate*(1.-to.inertiaCoefficient) ;
			*nnPrevUpdate *= to.inertiaCoefficient;
			*nnUpdate += (*nnPrevUpdate);
		} else {
			*nnUpdate *= -to.trainingRate;
		}
		*NN += *nnUpdate;	// apply changes
	
		Numerical newLoss = this->Eval(trainSet) + to.regulationCoefficient*this->RegulationPenalty(to.regulationType);		// test changes
		improvement = newLoss < Loss;
		
		if(!improvement) {			// reverse changes if worse
			std::cout << " didn't improve with trainingRate " << to.trainingRate << " because " << newLoss << " > " << Loss << std::endl;
			*nnUpdate *= (-1.);
			*NN += *nnUpdate;
			runImprovement = (runImprovement > 0 ? 0 : runImprovement-1);
		} else {
			Loss = newLoss;
			nnPrevUpdate = nnUpdate;
			runImprovement = (runImprovement < 0 ? 0 : runImprovement+1);
		}
		
		if(!improvement && to.useInertia)	// decrease impact of inertia if no improvement
			*nnPrevUpdate *= 1./2.; 
		
		if(!improvement && to.adjustTrainingRate){	// make trainingRate smaller 
			to.trainingRate/= 2.;
			to.trainingRate =std::max(to.trainingRate, to.trainingRateMin);
			to.trainingRate =std::max(to.trainingRate, std::numeric_limits<Numerical>::min());
		}
		if(to.adjustTrainingRate && to.trainingRate < Options.trainingRate && runImprovement > 5)	// slowly increase trainingRate if it runs okay and has been decreased before
			to.trainingRate *= 1.5;
			
			
		run++;
	}
	
	return run;
}

std::shared_ptr<Network> Trainer::Batch(const std::vector<DataPoint>& batch) const
{
	assert(batch.size() > 0);
	
	//Numerical prevLoss = this->Eval(batch);
	//std::cout << "start batch with loss " << prevLoss << std::endl;
	auto NNcp = std::make_shared<Network>(*NN);
	for(int i = 0; i < NNcp->numLayers(); i++)
		NNcp->layers.at(i).InitializeZero();
	
	for(uint i = 0; i < batch.size(); i++) {
		this->Backpropagation(batch.at(i), NNcp);
	}
	
	return NNcp;
}

}
