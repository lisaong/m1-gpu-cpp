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

    void reduce1D(MTL::Buffer *x_array,
                  MTL::Buffer *result_array,
                  size_t arrayLength);

private:
    void _blocking1D(std::vector<MTL::Buffer *> buffers,
                    size_t arrayLength,
                    const char *method);

    // The kernel function pipelines.
    std::map<std::string, AutoPtr<MTL::ComputePipelineState>> _mfunctionPipelineMap;

    // The command queue used to pass commands to the device.
    AutoPtr<MTL::CommandQueue> _mCommandQueue;
};