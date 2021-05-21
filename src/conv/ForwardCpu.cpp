// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "ForwardCpu.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

ForwardCpu::ForwardCpu(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim)
    {
}
VIRTUAL void ForwardCpu::forward(int batchSize, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper) {
    inputDataWrapper->copyToHost();
    weightsWrapper->copyToHost();
    float *bias = 0;
    if(dim.biased) {
        biasWrapper->copyToHost();
        bias =  (float *)biasWrapper->getHostArray();
    }
    float *output = forward(batchSize, (float *)inputDataWrapper->getHostArray(), (float *)weightsWrapper->getHostArray(), bias);
    int outputNumElements = batchSize * dim.outputCubeSize;
    float *hostArray = (float *)outputWrapper->getHostArray();
    for(int i = 0; i < outputNumElements; i++) {
        hostArray[i] = output[i];
    }
    outputWrapper->copyToDevice();
    delete[] output;
}
VIRTUAL float *ForwardCpu::forward(int batchSize, float *inputData, float *weights, float *bias) {
//    cout << "ForwardCpu::forward outputcubesize=" << dim.outputCubeSize << " batchSize=" << batchSize << endl;
    float *output = new float[ dim.outputCubeSize * batchSize ];
    for(int n = 0; n < batchSize; n++) {
        for(int filter = 0; filter < dim.numFilters; filter++) {
            for(int outRow = 0; outRow < dim.outputSize.height; outRow += 1 + dim.skip) {
                for(int outCol = 0; outCol < dim.outputSize.width; outCol += 1 + dim.skip) {
                    float sum = 0;
                    for(int inPlane = 0; inPlane < dim.inputPlanes; inPlane++) {
//                        cout << "inplane=" << inPlane << endl;
                        for(int u = -dim.halfFilterSize.height; u <= dim.halfFilterSize.height - dim.isEven; u++) {
                            int inRow = outRow * (dim.skip + 1) + u + (dim.padZeros ? 0 : dim.halfFilterSize.height);
//                                cout << "candidate inRow " << inRow << endl;
                            if(inRow < 0 || inRow > dim.inputSize.height - 1) {
                                continue;
                            }
                            int filterRow = u + dim.halfFilterSize.height;
                            for(int v = -dim.halfFilterSize.width; v <= dim.halfFilterSize.width - dim.isEven; v++) {
                                int inCol = outCol * (dim.skip + 1) + v + (dim.padZeros ? 0 : dim.halfFilterSize.width);
                                int filterCol = v + dim.halfFilterSize.width;
                                if(inCol < 0 || inCol > dim.inputSize.width - 1) {
                                    continue;
                                }
                                int inputIndex = (( n
                                    * dim.inputPlanes + inPlane)
                                    * dim.inputSize.height + inRow)
                                    * dim.inputSize.width + inCol;
                                int weightIndex = (( filter 
                                    * dim.inputPlanes + inPlane) 
                                    * dim.filterSize.height + filterRow)
                                    * dim.filterSize.width + filterCol;
//                                    cout << "inpos " << inRow << "," << inCol << " outpos " << outRow << "," << outCol
//                                        << " filterpos " << filterRow << "," << filterCol << endl;
                                float sumchange = inputData[ inputIndex] * weights[ weightIndex ];
                                if(sumchange != 0) {
//                                    std::cout << inputData[inputIndex] << " * " << weights[weightIndex] << " = " << sumchange << std::endl;
                                }
                                sum += sumchange;
//                                std::cout << "Weight[" << weightIndex << "] = " << weights[weightIndex] << std::endl;
//                               cout << "inputIndex=" << inputIndex << " weightIndex=" << weightIndex << 
//                                    "  inputData[inputIndex]=" << inputData[inputIndex] << " weights[weightIndex]=" << weights[weightIndex] << " sumchange " << sumchange << " sum=" << sum << endl;
                            }
                        }
                    }
                    if(dim.biased) {
                        sum += bias[filter];
                    }
//                    sum = fn->calc(sum);
                    int outputIndex = (( n 
                        * dim.numFilters + filter) 
                        * dim.outputSize.height + outRow)
                        * dim.outputSize.width + outCol;
                    output[outputIndex] = sum;
 //                   cout << "outputIndex=" << outputIndex << " sum=" << sum << " output[outputIndex]=" << output[outputIndex] << endl;
                }
            }
        }
    }
    return output;
}


