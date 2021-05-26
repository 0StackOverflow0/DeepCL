// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>

#include "util/StatefulTimer.h"
#include "util/Timer.h"
#include "net/NeuralNet.h"
#include "net/Trainable.h"
#include "NetAction.h"
#include "util/stringhelper.h"
#include "NetLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI NetLearner::NetLearner(Trainer *trainer, Trainable *net,
        int Ntrain, float *trainData, int *trainLabels,
        int Ntest, float *testData, int *testLabels,
        int batchSize) :
        net(net)
        {
//    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    dumpTimings = false;
    learningDone = false;

    this->trainData = new InputData(net->getInputCubeSize(), trainData);
    this->trainLabels = new LabeledData(net, trainLabels);
    this->testData = new InputData(net->getInputCubeSize(), testData);
    this->testLabels = new LabeledData(net, testLabels);

    trainBatcher = new LearnBatcher2(net, trainer, batchSize, Ntrain,
        this->trainData,
        this->trainLabels);
    testBatcher = new ForwardBatcher2(net, batchSize, Ntest,
        this->testData,
        this->testLabels);
}
PUBLICAPI NetLearner::NetLearner(Trainer* trainer, Trainable* net,
    int Ntrain, float* trainData, float* trainLabels,
    int Ntest, float* testData, float* testLabels,
    int batchSize) :
    net(net)
{
    //    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    dumpTimings = false;
    learningDone = false;

    this->trainData = new InputData(net->getInputCubeSize(), trainData);
    this->trainLabels = new ExpectedData(net, trainLabels);
    this->testData = new InputData(net->getInputCubeSize(), testData);
    this->testLabels = new ExpectedData(net, testLabels);

    trainBatcher = new LearnBatcher2(net, trainer, batchSize, Ntrain,
        this->trainData,
        this->trainLabels);
    testBatcher = new ForwardBatcher2(net, batchSize, Ntest,
        this->trainData,
        this->trainLabels);
}
VIRTUAL NetLearner::~NetLearner() {
    delete trainBatcher;
    delete testBatcher;
    delete trainData;
    delete trainLabels;
    delete testData;
    delete testLabels;
}
VIRTUAL void NetLearner::setSchedule(int numEpochs) {
    setSchedule(numEpochs, 0);
}
VIRTUAL void NetLearner::setDumpTimings(bool dumpTimings) {
    this->dumpTimings = dumpTimings;
}
VIRTUAL void NetLearner::setSchedule(int numEpochs, int nextEpoch) {
    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}
PUBLICAPI VIRTUAL void NetLearner::reset() {
//    cout << "NetLearner::reset()" << endl;
    learningDone = false;
    nextEpoch = 0;
//    net->setTraining(true);
    trainBatcher->reset();
    testBatcher->reset();
    timer.lap();
}
VIRTUAL void NetLearner::postEpochTesting() {
    if(dumpTimings) {
        StatefulTimer::dump(true);
    }
    float const* output = net->getOutput();
    vector<float> vOutput(output, output + net->getOutputNumElements());
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch+1));
//    cout << "annealed learning rate: " << trainBatcher->getLearningRate() <<
    cout << " training loss: " << trainBatcher->getEpochLoss() << endl;
    cout << " train accuracy: " << trainBatcher->getEpochNumRight() << "/" << trainBatcher->getN() << " " << (trainBatcher->getEpochNumRight() * 100.0f/ trainBatcher->getN()) << "%" << std::endl;
    net->setTraining(false);
    testBatcher->run(nextEpoch);
    cout << "test accuracy: " << testBatcher->getEpochNumRight() << "/" << testBatcher->getN() << " " << 
        (testBatcher->getEpochNumRight() * 100.0f / testBatcher->getN()) << "%" << endl;
    timer.timeCheck("after tests");
}
PUBLICAPI VIRTUAL bool NetLearner::tickBatch() { // just tick one learn batch, once all done, then run testing etc
//    int epoch = nextEpoch;
//    trainBatcher->setLearningRate(learningRate * pow(annealLearningRate, epoch) );
    net->setTraining(true);
    trainBatcher->tick(nextEpoch);       // returns false once all learning done (all epochs)
    if(trainBatcher->getEpochDone()) {
        postEpochTesting();
        nextEpoch++;
    }
//    cout << "check learningDone nextEpoch=" << nextEpoch << " numEpochs=" << numEpochs << endl;
    if(nextEpoch == numEpochs) {
//        cout << "setting learningdone to true" << endl;
        learningDone = true;
    }
    return !learningDone;
}
PUBLICAPI VIRTUAL bool NetLearner::getEpochDone() {
    return trainBatcher->getEpochDone();
}
PUBLICAPI VIRTUAL int NetLearner::getNextEpoch() {
    return nextEpoch;
}
PUBLICAPI VIRTUAL int NetLearner::getNextBatch() {
    return trainBatcher->getNextBatch();
}
PUBLICAPI VIRTUAL int NetLearner::getNTrain() {
    return trainBatcher->getN();
}
PUBLICAPI VIRTUAL int NetLearner::getBatchNumRight() {
    return trainBatcher->getEpochNumRight();
}
PUBLICAPI VIRTUAL float NetLearner::getBatchLoss() {
    return trainBatcher->getEpochLoss();
}
VIRTUAL void NetLearner::setBatchState(int nextBatch, int numRight, float loss) {
    trainBatcher->setBatchState(nextBatch, numRight, loss);
}
PUBLICAPI VIRTUAL bool NetLearner::tickEpoch() {
    if(trainBatcher->getEpochDone()) {
        trainBatcher->reset();
    }
    while(!trainBatcher->getEpochDone()) {
        tickBatch();
    }
    return !learningDone;
}
PUBLICAPI VIRTUAL void NetLearner::run() {
    if(learningDone) {
        reset();
    }
    while(!learningDone) {
        tickEpoch();
    }
}
PUBLICAPI VIRTUAL bool NetLearner::isLearningDone() {
    return learningDone;
}


