// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "LayerDimensions.h"

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT Backward {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
//    ActivationFunction const *upstreamFn;

    virtual ~Backward() {}
    virtual void backward( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutput, CLWrapper *weightsWrapper,
        CLWrapper *gradInput ) = 0;

    // [[[cog
    // import cog_addheaders    
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC Backward *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC Backward *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC Backward *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    Backward( OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL float * backward( int batchSize, float *input, float *gradOutput, float *filters );

    // [[[end]]]
};

