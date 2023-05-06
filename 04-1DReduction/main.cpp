#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "CPUOperations.hpp"
#include "MetalOperations.hpp"
#include "AutoPtr.hpp"

typedef std::chrono::microseconds time_unit;
constexpr auto unit_name = "microseconds";

// Configuration -----------------------------------------------------------------------
// Amount of repeats for benchmarking
constexpr size_t repeats = 100;
// Length of array to test kernels on
constexpr unsigned long arrayLength = 1 << 20;

constexpr unsigned long numThreadsPerGroup = 32;
constexpr unsigned long numThreadgroups = arrayLength / numThreadsPerGroup;
// end ---------------------------------------------------------------------------------

constexpr unsigned long bufferSize = arrayLength * sizeof(float);
constexpr unsigned long resultBufferSize = numThreadgroups * sizeof(float);

int main(int argc, char *argv[])
{
    AutoPtr<MTL::Device> device(MTL::CreateSystemDefaultDevice());

    std::cout << "Running on " << device->name()->utf8String() << std::endl;
    std::cout << "Array size " << arrayLength << ", tests repeated " << repeats
              << " times" << std::endl
              << std::endl;

    // TODO: profile managed vs. shared vs. device
    AutoPtr<MTL::Buffer> buf_MTL(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
    AutoPtr<MTL::Buffer> result_MTL(device->newBuffer(resultBufferSize, MTL::ResourceStorageModeManaged));

    // Get a C++-style reference to the buffer
    auto buf_CPP = reinterpret_cast<float *>(buf_MTL->contents());
    auto result_CPP = reinterpret_cast<float *>(result_MTL->contents());
    float result_GPU = 0.0f;
    float result_VER = 0.0f;

    generateRandomFloatData(buf_CPP, arrayLength);

    MetalOperations reductionOps(device.get());
    reductionOps.reduceSum1D(buf_MTL.get(), result_MTL.get(), arrayLength, "reduceSum1D_2", numThreadsPerGroup, numThreadgroups);

    reduce1D(buf_CPP, &result_VER, arrayLength);
    for (uint i = 0; i < numThreadgroups; ++i)
    {
        result_GPU += result_CPP[i];
    }

    if (result_VER == result_GPU)
    {
        std::cout << u8"\u2705" << "reduce1D: Metal and C++ results match" << std::endl;
    }
    else
    {
        std::cerr << u8"\u274C" << "reduce1D: Metal (" << result_GPU
                  << ") and C++ (" << result_VER
                  << ") results do not match" << std::endl;
        return -1;
    }

    return 0;
}