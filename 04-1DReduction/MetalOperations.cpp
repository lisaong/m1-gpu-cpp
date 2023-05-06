#include <iostream>
#include <algorithm>
#include "MetalOperations.hpp"

#define LIBRARY_PATH "./reduce.metallib"

MetalOperations::MetalOperations(MTL::Device *device) : _mDevice(device),
                                                        _mCommandQueue(_mDevice->newCommandQueue())
{
    NS::Error *error = nullptr;
    if (_mCommandQueue.get() == nullptr)
    {
        std::cerr << "Failed to find the command queue." << std::endl;
        return;
    }

    // Load the shader files with a .metal file extension in the project
    auto filepath = NS::String::string(LIBRARY_PATH, NS::ASCIIStringEncoding);
    AutoPtr<MTL::Library> opLibrary(_mDevice->newLibrary(filepath, &error));
    if (opLibrary.get() == nullptr)
    {
        std::cerr << "Failed to find the library at "
                  << LIBRARY_PATH << ", error"
                  << error->localizedDescription()->utf8String() << std::endl;
        return;
    }

    std::cout << "Available Metal functions in " << LIBRARY_PATH << std::endl;
    auto fnNames = opLibrary->functionNames();
    for (auto i = 0; i < fnNames->count(); ++i)
    {
        auto name_nsstring = fnNames->object(i)->description();
        auto name_utf8 = name_nsstring->utf8String();
        std::cout << name_utf8 << std::endl;

        // insert/find avoids calling the default constructor for AutoPtr
        _mfunctionPipelineMap.insert({name_utf8,
                                      AutoPtr<MTL::ComputePipelineState>(
                                          _mDevice->newComputePipelineState(
                                              opLibrary->newFunction(name_nsstring),
                                              &error))});

        if (auto found = _mfunctionPipelineMap.find(name_utf8); found == _mfunctionPipelineMap.end())
        {
            std::cerr << "Failed to create pipeline state object for "
                      << name_utf8 << ", error "
                      << error->localizedDescription()->utf8String() << std::endl;
            return;
        }
    }
    std::cout << std::endl;
}

const MTL::ComputePipelineState *MetalOperations::_getPipeline(const char *method)
{
    auto found = _mfunctionPipelineMap.find(method);
    if (found == _mfunctionPipelineMap.end())
    {
        std::cerr << "Failed to find pipeline state object for "
                  << method << std::endl;
        return nullptr;
    }

    return found->second.get();
}

void MetalOperations::reduceSum1D(MTL::Buffer *X,
                                  MTL::Buffer *result,
                                  unsigned long xLength,
                                  const char *method,
                                  unsigned long numThreadsPerGroup)
{
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();

    // Set the compute pipeline state.
    auto methodPSO = _getPipeline(method);
    if (methodPSO == nullptr)
    {
        return;
    }
    computeEncoder->setComputePipelineState(methodPSO);

    const auto gridSize = MTL::Size::Make(xLength, 1, 1);
    const auto threadgroupSize = MTL::Size::Make(numThreadsPerGroup, 1, 1);
    const auto sharedMemSize = numThreadsPerGroup * sizeof(float);

    // Set the buffer to be used as the compute shader's input.
    computeEncoder->setBuffer(X, 0, 0);
    computeEncoder->setBuffer(result, 0, 1);

    // Set the threadgroup shared buffer size.
    computeEncoder->setThreadgroupMemoryLength(sharedMemSize, 0);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}