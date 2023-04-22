#include "CPUOperations.hpp"
#include <math.h>

void generateRandomFloatData(float *dataPtr, unsigned long arrayLength)
{
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}
