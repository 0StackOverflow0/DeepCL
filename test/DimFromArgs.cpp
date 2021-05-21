// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>

#include "conv/LayerDimensions.h"
#include "test/DeepCLGtestGlobals.h"
#include "test/TestArgsParser.h"

#include "DimFromArgs.h"

using namespace std;

void DimFromArgs::arg( LayerDimensions *p_dim ) {
    TestArgsParser::arg( "inputplanes", &(p_dim->inputPlanes) );
    TestArgsParser::arg( "numinputplanes", &(p_dim->inputPlanes) );
    TestArgsParser::arg( "inputheight", &(p_dim->inputSize.height) );
    TestArgsParser::arg("inputwidth", &(p_dim->inputSize.width));
    TestArgsParser::arg( "numfilters", &(p_dim->numFilters) );
    TestArgsParser::arg( "filterheight", &(p_dim->filterSize.height) );
    TestArgsParser::arg("filterwidth", &(p_dim->filterSize.width));
    TestArgsParser::arg( "padzeros", &(p_dim->padZeros) );
    TestArgsParser::arg( "biased", &(p_dim->biased) );
//    cout << "DimFromArgs::arg() " << *p_dim << endl;
}


