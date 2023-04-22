#include <iostream>
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
        _mfunctionPipelineMap.insert({name_utf8, AutoPtr<MTL::ComputePipelineState>(
                                                     _mDevice->newComputePipelineState(opLibrary->newFunction(name_nsstring), &error))});

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

MTL::Buffer *MetalOperations::_reduceSum1D_threadgroup(MTL::Buffer *X,
                                                       unsigned long xLength,
                                                       unsigned long *numThreadGroups)
{
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();

    // Set the compute pipeline state.
    auto methodPSO = _getPipeline("reduceSum1D_threadgroup");
    if (methodPSO == nullptr)
    {
        return nullptr;
    }
    computeEncoder->setComputePipelineState(methodPSO);

    auto gridSize = MTL::Size::Make(xLength, 1, 1);

    // Calculate a threadgroup size.
    auto threadGroupSize = methodPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > xLength)
    {
        threadGroupSize = xLength;
    }
    auto threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    AutoPtr<MTL::Buffer> results(_mDevice->newBuffer(threadGroupSize * sizeof(float),
                                                     MTL::ResourceStorageModeManaged));

    // Set the buffer to be used as the compute shader's input.
    computeEncoder->setBuffer(X, 0, 0);
    computeEncoder->setBuffer(results.get(), 0, 1);

    // Set the threadgroup shared buffer size.
    computeEncoder->setThreadgroupMemoryLength(sizeof(float) * threadGroupSize, 0);

    // Encode the compute command.
    computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);

    // End the compute command encoder.
    computeEncoder->endEncoding();

    // Commit the command buffer.
    commandBuffer->commit();

    *numThreadGroups = threadGroupSize;
    return results.get();
}

void MetalOperations::_reduceSum1D_final(MTL::Buffer *threadGroupSums,
                                         unsigned long numThreadGroups,
                                         MTL::Buffer *result)
{
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();

    // Set the compute pipeline state.
    auto methodPSO = _getPipeline("reduceSum1D_final");
    if (methodPSO == nullptr)
    {
        return;
    }
    computeEncoder->setComputePipelineState(methodPSO);

    auto gridSize = MTL::Size::Make(numThreadGroups, 1, 1);

    // Calculate a threadgroup size.
    auto threadGroupSize = methodPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > numThreadGroups)
    {
        threadGroupSize = numThreadGroups;
    }
    auto threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Set the buffer to be used as the compute shader's input.
    computeEncoder->setBuffer(threadGroupSums, 0, 0);
    computeEncoder->setBuffer(result, 0, 1);

    // Set the threadgroup shared buffer size.
    computeEncoder->setThreadgroupMemoryLength(sizeof(float) * threadGroupSize, 0);

    computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
}

void MetalOperations::reduceSum1D(MTL::Buffer *X,
                                  MTL::Buffer *result,
                                  unsigned long xLength)
{
    unsigned long numThreadGroups;
    AutoPtr<MTL::Buffer> threadgroupResults(_reduceSum1D_threadgroup(X, xLength, &numThreadGroups));
    if (threadgroupResults.get() == nullptr)
    {
        return;
    }
    _reduceSum1D_final(threadgroupResults.get(), numThreadGroups, result);
}