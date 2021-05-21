// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <cstring>

#include "PatchExtractor.h"

using namespace std;

void PatchExtractor::extractPatch(int n, int numPlanes, Dimensions imageSize, Dimensions patchSize, int patchRow, int patchCol, float *source, float *destination) {
//    int n = 0;
    for(int plane = 0; plane < numPlanes; plane++) {
        float *upstreamImage = source + (n * numPlanes + plane) * imageSize.height * imageSize.width;
        float *outputImage = destination + (n * numPlanes + plane) * patchSize.height * patchSize.width;
        for(int outRow = 0; outRow < patchSize.height; outRow++) {
            const int inRow = outRow + patchRow;
            memcpy(&(outputImage[ outRow * patchSize.width ]), 
                &(upstreamImage[ inRow * imageSize.width + patchCol ]),
                patchSize.width * sizeof(float));
        }        
    }
}


