#include "CPUOperations.hpp"
#include <math.h>

void generateRandomFloatData(float *dataPtr, unsigned long arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; ++index)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

void reduce1D(const float *x, float *result, unsigned long arrayLength)
{
    *result = x[0];
    for (unsigned long index = 1; index < arrayLength; ++index)
    {
        *result += x[index];
    }
}