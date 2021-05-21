#pragma once

#if defined(_WIN32) 
# if defined(DeepCL_EXPORTS)
#  define DeepCL_EXPORT __declspec(dllexport)
# else
#  define DeepCL_EXPORT __declspec(dllimport)
# endif // DeepCL_EXPORTS
#else // _WIN32
# define DeepCL_EXPORT
#endif

// does nothing, just a marker, means it is part of
// our semantic versioning 'stable' api
#define PUBLICAPI

#define PUBLIC
#define PROTECTED
#define PRIVATE

typedef unsigned char uchar;

typedef long long int64;
typedef int int32;

class DeepCL_EXPORT Dimensions {
public:
    int width;
    int height;
    bool single;
    Dimensions() :
        width(0),
        height(0),
        single(false) {};
    Dimensions(int width, int height=-1) :
        width(width),
        height(height >= 0 ? height : width),
        single( width != height && height == 1 ) {};
    Dimensions operator+(const int& b) {
        return Dimensions(this->width + b, this->height + b); //this->height > 1 ? this->height + b : 1);
    };
    Dimensions operator+(const Dimensions& b) {
        return Dimensions(this->width + b.width, this->height + b.height); //this->height > 1 ? this->height + b : 1);
    };
    Dimensions operator-(const Dimensions& b) const {
        return Dimensions(this->width - b.width, this->height - b.height); // this->height > 1 ? this->height - b.height : 1);
    };
    Dimensions operator*(const Dimensions& b) {
        return Dimensions(this->width * b.width, this->height * b.height);
    }
    Dimensions operator/(const int& b) {
        return Dimensions(this->width / b, this->single ? 1 : this->height / b); // this->height > 1 ? this->height / b : 1);
    };
    //Dimensions operator=(const Dimensions& b) {
    //    return Dimensions(b.width, b.height);
    //}
    bool operator>(const Dimensions& b) {
        return this->width > b.width || this->height > b.height;
    }
};

extern bool operator==(const Dimensions& a, const int& b);

extern bool operator==(const Dimensions& a, const Dimensions& b);

extern bool operator!=(const Dimensions& a, const Dimensions& b);
