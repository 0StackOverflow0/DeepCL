#include <iostream>
#include <ostream>

#include "util/stringhelper.h"

#include "conv/LayerDimensions.h"

using namespace std;

ostream &operator<<(ostream &os, const LayerDimensions &dim) {
    os << "LayerDimensions{";
    os << " inputPlanes=" << dim.inputPlanes;
    os << " inputWidth=" << dim.inputSize.width;
    os << " inputHeight=" << dim.inputSize.height;
    os << " numFilters=" << dim.numFilters;
    os << " filterWidth=" << dim.filterSize.width;
    os << " filterHeight=" << dim.filterSize.height;
    os << " outputWidth=" << dim.outputSize.width;
    os << " outputHeight=" << dim.outputSize.height;
    os << " padZeros=" << dim.padZeros;
    os << " biased=" << dim.biased;
    os << " skip=" << dim.skip;
    os << "}";
    return os;
}

void LayerDimensions::deriveOthers() {
    this->numInputPlanes = inputPlanes;
    this->isEven = filterSize.height % 2 == 0 && filterSize.width % 2 == 0 ? 1 : 0;

    if (padZeros) {
        this->outputSize = this->isEven ? (inputSize / (skip + 1)) + 1 : inputSize / (skip + 1);
    } else {
        this->outputSize = (inputSize - filterSize / (skip + 1)) + 1;
    }
            
    this->inputSizeSquared = inputSize.height * inputSize.width;
    this->filterSizeSquared = filterSize.height * filterSize.width;
    this->outputSizeSquared = outputSize.height * outputSize.width;

    this->inputCubeSize = inputPlanes * inputSizeSquared;
    this->filtersSize = inputPlanes * numFilters * filterSizeSquared;
    this->outputCubeSize = numFilters * outputSizeSquared;
    this->halfFilterSize = Dimensions(filterSize.width / 2, filterSize.height / 2);
//    cout << "deriveOthers()" << *this << endl;
}

string LayerDimensions::buildOptionsString() {
    string options = "";
    if(biased) {
         options += " -D BIASED";
    }
    options += " -D gNumInputPlanes=" + toString(inputPlanes);
    options += " -D gInputPlanes=" + toString(inputPlanes);
    options += " -D gInputWidth=" + toString(inputSize.width);
    options += " -D gInputHeight=" + toString(inputSize.height);
    options += " -D gInputArea=" + toString(this->inputSizeSquared);
    options += " -D gNumFilters=" + toString(numFilters);
    options += " -D gFilterWidth=" + toString(filterSize.width);
    options += " -D gFilterHeight=" + toString(filterSize.height);
    options += " -D gHalfFilterWidth=" + toString(filterSize.width / 2);
    options += " -D gHalfFilterHeight=" + toString(filterSize.height / 2);
    options += " -D gFilterArea=" + toString(this->filterSizeSquared);
    options += " -D gNumOutputPlanes=" + toString(numFilters);
    options += " -D gOutputPlanes=" + toString(numFilters);
    options += " -D gOutputWidth=" + toString(outputSize.width);
    options += " -D gOutputHeight=" + toString(outputSize.height);
    options += " -D gOutputArea=" + toString(this->outputSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gMarginWidth=" + toString(padZeros ? this->halfFilterSize.width : 0);
    options += " -D gMarginHeight=" + toString(padZeros ? this->halfFilterSize.height : 0);
    options += " -D gEven=" + toString(filterSize.width % 2 == 0 && filterSize.height % 2 == 0 ? 1 : 0);
    options += " -D gSkip=" + toString(skip);
    return options;
}

