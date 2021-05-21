
#include "DeepCLDllExport.h"

bool operator==(const Dimensions& a, const int& b) {
    return a.width == b;
}

bool operator==(const Dimensions& a, const Dimensions& b) {
    return a.width == b.width && a.height == b.height;
}

bool operator!=(const Dimensions& a, const Dimensions& b) {
    return !(a == b);
}