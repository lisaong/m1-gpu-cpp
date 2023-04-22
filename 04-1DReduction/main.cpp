#include <iostream>
#include <assert.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
// #include "QuartzCore/QuartzCore.hpp"

typedef std::chrono::microseconds time_unit;
const auto unit_name = "microseconds";

// Configuration -----------------------------------------------------------------------
// Amount of repeats for benchmarking
const size_t repeats = 100;
// Length of array to test kernels on

const unsigned int arrayLength = 60 * 180 * 10000;
// end ---------------------------------------------------------------------------------

const unsigned int bufferSize = arrayLength * sizeof(float);

int main(int argc, char *argv[])
{
    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    std::cout << "Running on " << device->name()->utf8String() << std::endl;
    std::cout << "Array size " << arrayLength << ", tests repeated " << repeats
              << " times" << std::endl
              << std::endl;
}