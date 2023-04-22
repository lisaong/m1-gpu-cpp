#pragma once

#include <map>
#include <string>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#include "AutoPtr.hpp"

class MetalOperations
{
public:
    MTL::Device *_mDevice;
    MetalOperations(MTL::Device *device);
    ~MetalOperations() = default;

    void reduceSum1D(MTL::Buffer *X,
                     MTL::Buffer *result,
                     unsigned long xLength);

private:
    MTL::Buffer *_reduceSum1D_threadgroup(MTL::Buffer *X,
                                          unsigned long xLength,
                                          unsigned long *numThreadGroups);

    void _reduceSum1D_final(MTL::Buffer *threadGroupSums,
                            unsigned long numThreadGroups,
                            MTL::Buffer *result);

    const MTL::ComputePipelineState *_getPipeline(const char *method);

    // The kernel function pipelines.
    std::map<std::string, AutoPtr<MTL::ComputePipelineState>> _mfunctionPipelineMap;

    // The command queue used to pass commands to the device.
    AutoPtr<MTL::CommandQueue> _mCommandQueue;
};