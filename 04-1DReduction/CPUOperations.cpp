#include "CPUOperations.hpp"
#include <math.h>

void generateRandomData(int *dataPtr, unsigned long arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; ++index)
    {
        dataPtr[index] = rand() % 256 - 128;
    }
}

void reduceSum1D(const int *x, long *result, unsigned long arrayLength)
{
    *result = x[0];
    for (unsigned long index = 1; index < arrayLength; ++index)
    {
        *result += x[index];
    }
}