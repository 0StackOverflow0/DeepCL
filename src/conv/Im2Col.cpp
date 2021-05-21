//#include "clblas/ClBlasInstance.h"
#include "clblas/ClBlasHelper.h"
#include "EasyCL.h"
#include "templates/TemplatedKernel.h"

#include "Im2Col.h"

#include <iostream>
#include <stdexcept>
using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL
#define PUBLIC

PUBLIC Im2Col::Im2Col(EasyCL *cl, LayerDimensions dim) :
        cl(cl),
        dim(dim) {
//    ClBlasInstance::initializeIfNecessary();
    this->kernelIm2Col = 0;
    this->kernelCol2Im = 0;
}
PUBLIC VIRTUAL Im2Col::~Im2Col() {
    delete kernelIm2Col;
    delete kernelCol2Im;
}
void Im2Col::setupBuilder(TemplatedKernel *builder) {
    Dimensions size = dim.inputSize;
    Dimensions padding = dim.padZeros ? dim.halfFilterSize : 0;
    int stride = 1;
    int channels = dim.inputPlanes;
    Dimensions size_col = ( (size + ( padding * 2 ) - dim.filterSize ) / stride ) + 1;

    this->numKernelsIm2Col = channels * size_col.height * size_col.width;
    this->numKernelsCol2Im = channels * dim.inputSizeSquared;

    builder->set("paddingWidth", dim.padZeros ? dim.halfFilterSize.width : 0);
    builder->set("paddingHeight", dim.padZeros ? dim.halfFilterSize.height : 0);
    builder->set("stride", 1);
    builder->set("colWidth", size_col.width);
    builder->set("colHeight", size_col.height);
    builder->set("channels", dim.inputPlanes);
    builder->set("filterWidth", dim.filterSize.width);
    builder->set("filterHeight", dim.filterSize.height);
    builder->set("imWidth", dim.inputSize.width);
    builder->set("imHeight", dim.inputSize.height);
}
void Im2Col::buildKernelIm2Col() {
    TemplatedKernel builder(cl);
    setupBuilder(&builder);
    this->kernelIm2Col = builder.buildKernel(
        "im2col",
        "ForwardIm2Col.cl",
        getKernelTemplate(),
        "im2col",
        false
    );
}
void Im2Col::buildKernelCol2Im() {
    TemplatedKernel builder(cl);
    setupBuilder(&builder);
    this->kernelCol2Im = builder.buildKernel(
        "col2im",
        "ForwardIm2Col.cl",
        getKernelTemplate(),
        "col2im",
        false
    );
}
PUBLIC void Im2Col::im2Col(CLWrapper *imagesWrapper, int imagesOffset, CLWrapper *columnsWrapper) {
    if(kernelIm2Col == 0) {
        buildKernelIm2Col();
    }
    kernelIm2Col->in(numKernelsIm2Col);
    kernelIm2Col->in(imagesWrapper);
    kernelIm2Col->in(imagesOffset);
    kernelIm2Col->out(columnsWrapper);

    int workgroupSize = cl->getMaxWorkgroupSize();
    int numWorkgroups = this->numKernelsIm2Col;

    kernelIm2Col->run_1d(numWorkgroups * workgroupSize, workgroupSize);
}
PUBLIC void Im2Col::col2Im(CLWrapper *columnsWrapper, CLWrapper *imagesWrapper, int imagesOffset) {
    if(kernelCol2Im == 0) {
        buildKernelCol2Im();
    }
    kernelCol2Im->in(numKernelsCol2Im);
    kernelCol2Im->in(columnsWrapper);
    kernelCol2Im->out(imagesWrapper);
    kernelCol2Im->in(imagesOffset);

    int workgroupSize = cl->getMaxWorkgroupSize();
    int numWorkgroups = this->numKernelsCol2Im;

//        cout << "numworkgroups=" << numWorkgroups << " workgorupSize=" << workgroupSize << endl;
    kernelCol2Im->run_1d(numWorkgroups * workgroupSize, workgroupSize);
}
STATIC std::string Im2Col::getKernelTemplate() {
    // [[[cog
    // import stringify
    // stringify.write_kernel("kernel", "cl/ForwardIm2Col.cl")
    // ]]]
    // generated using cog, from cl/ForwardIm2Col.cl:
    const char * kernelSource =  
    "// from SpatialConvolutionMM.cu:\n"
    "\n"
    "// CL: grid stride looping\n"
    "#define CL_KERNEL_LOOP(i, n)                        \\\n"
    "  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\\n"
    "      i < (n);                                       \\\n"
    "      i += get_local_size(0) * get_num_groups(0))\n"
    "\n"
    "//#define gPadding {{padding}}\n"
    "//#define gStride {{stride}}\n"
    "//#define gColSize {{colSize}}\n"
    "//#define gFilterSize {{filterSize}}\n"
    "//#define gSize {{size}}\n"
    "\n"
    "// Kernel for fast unfold+copy\n"
    "// (adapted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)\n"
    "kernel void im2col(\n"
    "    const int n,\n"
    "    global float const * im_data, int im_offset,\n"
    "    global float* data_col) {\n"
    "  global const float *data_im = im_data + im_offset;\n"
    "\n"
    "  CL_KERNEL_LOOP(index, n) {\n"
    "    int w_out = index % {{colWidth}};\n"
    "    index /= {{colWidth}};\n"
    "    int h_out = index % {{colHeight}};\n"
    "    int channel_in = index / {{colWidth}};\n"
    "    int channel_out = channel_in * {{filterHeight}} * {{filterWidth}};\n"
    "    int h_in = h_out * {{stride}} - {{paddingHeight}};\n"
    "    int w_in = w_out * {{stride}} - {{paddingWidth}};\n"
    "    data_col += (channel_out * {{colHeight}} + h_out) * {{colWidth}} + w_out;\n"
    "    data_im += (channel_in * {{imHeight}} + h_in) * {{imWidth}} + w_in;\n"
    "    for (int i = 0; i < {{filterHeight}}; ++i) {\n"
    "      for (int j = 0; j < {{filterWidth}}; ++j) {\n"
    "        int h = h_in + i;\n"
    "        int w = w_in + j;\n"
    "        *data_col = (h >= 0 && w >= 0 && h < {{imHeight}} && w < {{imWidth}}) ?\n"
    "          data_im[i * {{imWidth}} + j] : 0;\n"
    "        data_col += {{colHeight}} * {{colWidth}};\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "\n"
    "kernel void col2im(\n"
    "    const int n,\n"
    "    global float const *data_col,\n"
    "    global float* im_data, int im_offset) {\n"
    "  global float *data_im = im_data + im_offset;\n"
    "\n"
    "  for (int index = get_group_id(0) * get_local_size(0) + get_local_id(0); index < (n); index += get_local_size(0) * get_num_groups(0)) {\n"
    "    float val = 0;\n"
    "    int w = index % {{imWidth}} + {{paddingWidth}};\n"
    "    int h = (index / {{imWidth}}) % {{imHeight}} + {{paddingHeight}};\n"
    "    int c = index / ({{imHeight}} * {{imWidth}});\n"
    "    // compute the start and end of the output\n"
    "    int w_col_start = (w < {{filterWidth}}) ? 0 : (w - {{filterWidth}}) / {{stride}} + 1;\n"
    "    int w_col_end = min(w / {{stride}} + 1, {{colWidth}});\n"
    "    int h_col_start = (h < {{filterHeight}}) ? 0 : (h - {{filterHeight}}) / {{stride}} + 1;\n"
    "    int h_col_end = min(h / {{stride}} + 1, {{colHeight}});\n"
    "\n"
    "    int offset = (c * {{filterWidth}} * {{filterHeight}} + h * {{filterWidth}} + w) * {{colHeight}} * {{colWidth}};\n"
    "    int coeff_h_col = (1 - {{stride}} * {{filterHeight}} * {{colHeight}}) * {{colWidth}};\n"
    "    int coeff_w_col = (1 - {{stride}} * {{colHeight}} * {{colWidth}});\n"
    "    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n"
    "      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n"
    "        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];\n"
    "      }\n"
    "    }\n"
    "    data_im[index] = val;\n"
    "  }\n"
    "}\n"
    "\n"
    "";
    // [[[end]]]
    return kernelSource;
}

