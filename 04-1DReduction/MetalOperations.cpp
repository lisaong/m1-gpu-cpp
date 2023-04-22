#include <iostream>
#include "MetalOperations.hpp"

#define LIBRARY_PATH "./reduce.metallib"

MetalOperations::MetalOperations(MTL::Device *device) :
    _mDevice(device),
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
        _mfunctionPipelineMap.insert({ name_utf8, AutoPtr<MTL::ComputePipelineState>(
            _mDevice->newComputePipelineState(opLibrary->newFunction(name_nsstring), &error)) });

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


void MetalOperations::_blocking1D(std::vector<MTL::Buffer *> buffers,
                                 size_t arrayLength,
                                 const char *method)
{
    MTL::CommandBuffer* commandBuffer = _mCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // Set the compute pipeline state.
    auto found = _mfunctionPipelineMap.find(method);
    if (found == _mfunctionPipelineMap.end())
    {
        std::cerr << "Failed to find pipeline state object for "
                  << method << std::endl;
        return;
    }

    auto methodPSO = found->second.get();
    computeEncoder->setComputePipelineState(methodPSO);

    // Set the buffer to be used as the compute shader's input.
    computeEncoder->setBuffer(buffers[0], 0, 0);
    computeEncoder->setBuffer(buffers[1], 0, 1);

    auto gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    auto threadGroupSize = methodPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    auto threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);

    // End the compute command encoder.
    computeEncoder->endEncoding();

    // Commit the command buffer.
    commandBuffer->commit();
}


void MetalOperations::reduce1D(MTL::Buffer *x_array,
                               MTL::Buffer *result_array,
                               size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {x_array,
                                          result_array};
    _blocking1D(buffers, arrayLength, "reduce1D");
}