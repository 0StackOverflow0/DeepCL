// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>

#include "DeepCL.h"
#include "AccuracyHelper.h"
#include "conv/Forward.h"

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

TEST(testDeepCL, basic) {
    DeepCL *cl = DeepCL::createForFirstGpuOtherwiseCpu();

    int batchSize = 2;
    int numInPlanes = 1; int imageSize = 2;
    int numOutPlanes = 2; int filterWidth = 2;
    int padZeros = 0;
    float data[] = { 0, 0, 
                      0.5f, 0.5f,

                        13, 17,
                       -19, 2.3f,
};
    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,
 };
    int resultSize = 4;
    float expectedOutput[] = {
        -0.5f * 0.5f + 0.5f * 0.5f,
        0.7f * 0.5f -1.1f * 0.5f,
        (-0.5f) * (-19) + 0.5f * 2.3f,
        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
    };
    cout << "expected number of output: " << resultSize << endl;
//    int outputSize = 0;
    for( int i = 1; i <= 4; i++ ) {
        Forward *forward = Forward::instanceSpecific( 3, cl,
            LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
            padZeros == 1, false ) );
        float *output = new float[forward->getOutputTotalSize(batchSize)];
        forward->forward( batchSize, data, filter1, 0, output );  
        for( int result = 0; result < resultSize; result++ ) {
            ASSERT_EQ( expectedOutput[result], output[result] );
        }
        delete forward;
        delete[] output;
    }

    delete cl;
}

int numPlanes = 9;
Dimensions imageSize(128, 1);

int learningRate = 0.02f;

string filepath = "..\\data\\uci har\\";

vector<string> groups({ "train", "test" });

string labelname = "\\y_";

string midpath = "\\Inertial Signals\\";

vector<string> names({ "body_acc_x_",  "body_acc_y_",  "body_acc_z_",
                        "body_gyro_x_", "body_gyro_x_", "body_gyro_x_",
                        "total_acc_x_", "total_acc_x_", "total_acc_x_" });

string extension = ".txt";

vector<int> getLabels(bool train = true) {
    vector<int> result;
    size_t groupIndex = train ? 0 : 1;
    stringstream fullpath;
    fullpath << filepath << groups[groupIndex] << labelname << groups[groupIndex] << extension;

    ifstream labels(fullpath.str(), fstream::in);

    string line;
    while (getline(labels, line)) {
        if (line.size() > 0) {
            int label = atoi(line.c_str());
            result.push_back(label);
        }
    }
    return result;
}

vector<float> getData(string fullpath) {
    vector<float> result;
    ifstream file(fullpath, fstream::in);

    string line, cell;
    while (getline(file, line)) {
        if (line.size() > 0) {
            stringstream row(line);
            size_t count = 0;
            while (getline(row, cell, ' '))
                if (cell.size() > 0) {
                    result.push_back(atof(cell.c_str()));
                    ++count;
                }
            if (count != 128)
                cout << count << endl;
        }
    }

    return result;
}

vector<float> getData(int examples, bool train = true) {
    vector<float> result;
    vector<vector<float>> files;

    // Collect file data
    for (auto name : names) {
        size_t groupIndex = train ? 0 : 1;
        stringstream fullpath;
        fullpath << filepath << groups[groupIndex] << midpath << name << groups[groupIndex] << extension;
        files.push_back(getData(fullpath.str()));
    }

    // Check Integrity
    for (auto file : files) {
        if (examples != file.size() / 128)
            // Error, mismatch lengths
            return result;
    }

    // Interlace Planes by Example
    for (auto example = 0; example < examples; ++example)
        for (auto file : files)
            for (auto index = 0; index < 128; ++index)
                result.push_back(file[example * 128 + index]);

    return result;
}

void buildNet(NeuralNet* net, size_t batchSize) {

    net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
    net->addLayer(ConvolutionalMaker::instance()->numFilters(64)->filterSize(Dimensions(3,1)));
    net->addLayer(ActivationMaker::instance()->relu());
    net->addLayer(ConvolutionalMaker::instance()->numFilters(64)->filterSize(Dimensions(3,1)));
    net->addLayer(ActivationMaker::instance()->relu());
    net->addLayer(DropoutMaker::instance());
    net->addLayer(PoolingMaker::instance());
    net->addLayer(FullyConnectedMaker::instance()->numPlanes(100)->imageSize(1)->weightsInitializer(new OriginalInitializer()));
    net->addLayer(ActivationMaker::instance()->relu());
    net->addLayer(FullyConnectedMaker::instance()->numPlanes(7)->imageSize(1)->weightsInitializer(new OriginalInitializer()));
    net->addLayer(SoftMaxMaker::instance());

    net->setBatchSize(batchSize);
    net->print();
}

SGD* getSGD(EasyCL* cl) {
    SGD * sgd = new SGD(cl);
    sgd->setLearningRate(learningRate);
    sgd->setMomentum(0.0f);
    sgd->setWeightDecay(0.0f);
    return sgd;
}

auto train(vector<int>& testLabels, vector<int>& trainLabels, vector<float>& testData, vector<float>& trainData) {
    unique_ptr<EasyCL> cl(EasyCL::createForFirstGpuOtherwiseCpu());

    size_t batchSize = 32;

    unique_ptr<NeuralNet> net(new NeuralNet(cl.get()));
    buildNet(net.get(), batchSize);

    SGD * sgd = getSGD(cl.get());
    unique_ptr<Trainer> trainer(sgd);

    unique_ptr<NetLearner> netLearner(new NetLearner(trainer.get(), net.get(),
        trainLabels.size(), trainData.data(), trainLabels.data(),
        testLabels.size(), testData.data(), testLabels.data(),
        batchSize
    ));

    netLearner->reset();
    netLearner->setSchedule(10, 0);

    while (!netLearner->isLearningDone())
        netLearner->tickBatch();

    return netLearner->trainBatcher->getLoss();
}

void save(vector<int> &trainLabels, vector<int> &testLabels, vector<float> &trainData, vector<float> &testData) {
    ofstream trainL("..\\data\\uci_har\\trainLabels.dat", fstream::out | fstream::binary);
    trainL.write(reinterpret_cast<char*>(trainLabels.data()), trainLabels.size() * sizeof(int));

    ofstream testL("..\\data\\uci_har\\testLabels.dat", fstream::out | fstream::binary);
    testL.write(reinterpret_cast<char*>(testLabels.data()), testLabels.size() * sizeof(int));

    ofstream trainD("..\\data\\uci_har\\trainData.dat", fstream::out | fstream::binary);
    trainD.write(reinterpret_cast<char*>(trainData.data()), trainData.size() * sizeof(float));

    ofstream testD("..\\data\\uci_har\\testData.dat", fstream::out | fstream::binary);
    testD.write(reinterpret_cast<char*>(testData.data()), testData.size() * sizeof(float));

    trainL.close();
    testL.close();
    trainD.close();
    testD.close();
}

auto readFresh() {
    vector<int> trainLabels(getLabels());
    vector<int> testLabels(getLabels(false));

    vector<float> trainData(getData(trainLabels.size()));
    vector<float> testData(getData(testLabels.size(), false));
    return std::tuple<vector<int>, vector<int>, vector<float>, vector<float>>( trainLabels, testLabels, trainData, testData );
}

auto readSaved() {
    size_t filesize = 0;

    ifstream trainL("..\\data\\uci_har\\trainLabels.dat", fstream::in | fstream::binary | fstream::ate);
    filesize = trainL.tellg();
    vector<int> trainLabels(filesize / sizeof(int));
    trainL.seekg(0);
    trainL.read(reinterpret_cast<char*>(trainLabels.data()), filesize);
    trainL.close();

    ifstream testL("..\\data\\uci_har\\testLabels.dat", fstream::in | fstream::binary | fstream::ate);
    filesize = testL.tellg();
    vector<int> testLabels(filesize / sizeof(int));
    testL.seekg(0);
    testL.read(reinterpret_cast<char*>(testLabels.data()), filesize);
    testL.close();

    ifstream trainD("..\\data\\uci_har\\trainData.dat", fstream::in | fstream::binary | fstream::ate);
    filesize = trainD.tellg();
    vector<float> trainData(filesize / sizeof(float));
    trainD.seekg(0);
    trainD.read(reinterpret_cast<char*>(trainData.data()), filesize);
    trainD.close();

    ifstream testD("..\\data\\uci_har\\testData.dat", fstream::in | fstream::binary | fstream::ate);
    filesize = testD.tellg();
    vector<float> testData(filesize / sizeof(float));
    testD.seekg(0);
    testD.read(reinterpret_cast<char*>(testData.data()), filesize);
    testD.close();

    return std::tuple<vector<int>, vector<int>, vector<float>, vector<float>>(trainLabels, testLabels, trainData, testData);
}

void process() {
    auto tuple = readFresh();

    auto trainLabels(std::get<0>(tuple));
    auto testLabels(std::get<1>(tuple));
    auto trainData(std::get<2>(tuple));
    auto testData(std::get<3>(tuple));

    save(trainLabels, testLabels, trainData, testData);
}

TEST(testDeepCL, uci_har) {
    process();

    auto tuple = readSaved();

    auto trainLabels = std::get<0>(tuple);
    auto testLabels = std::get<1>(tuple);
    auto trainData = std::get<2>(tuple);
    auto testData = std::get<3>(tuple);

    ASSERT_GE(0.01f, train(testLabels, trainLabels, testData, trainData));
}
