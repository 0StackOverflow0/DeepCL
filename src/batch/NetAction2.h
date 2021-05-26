// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DeepCLDllExport.h"

class Trainable;
class Trainer;
//class TraininerContext;

// #include "trainers/TrainingContext.h"
#include "batch/BatchData.h"

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// compared with NetAction objects, these objects use the InputData and OutputData
// abstractions, so can workt with both expected data, and labels, for example

class DeepCL_EXPORT NetAction2 {
public:
    float epochLoss;
    int epochNumRight;
    virtual ~NetAction2() {}
    virtual void run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData) = 0;
    void reset() {
        epochLoss = 0;
        epochNumRight = 0;
    }
    float getEpochLoss() {
        return epochLoss;
    }
    int getEpochNumRight() {
        return epochNumRight;
    }
};

class DeepCL_EXPORT NetLearnAction2 : public NetAction2 {
public:
    Trainer *trainer;
    NetLearnAction2(Trainer *trainer) :
        trainer(trainer) {
        epochLoss = 0;
        epochNumRight = 0;
    }   
    virtual void run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData);
};

class DeepCL_EXPORT NetForwardAction2 : public NetAction2 {
public:
    NetForwardAction2() {
        epochLoss = 0;
        epochNumRight = 0;
    }
    virtual void run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData);
};

