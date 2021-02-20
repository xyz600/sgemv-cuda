#include <cuda_runtime_api.h>

__global__ void sgemv_dev(const float *__restrict__ matrix, const float *__restrict__ vector, float *result,
                          const int size)
{
    constexpr int unroll_size = 4;
    const float *a = &matrix[(blockIdx.y * blockDim.y + threadIdx.y) * unroll_size * size];
    for (int i = (blockIdx.y * blockDim.y + threadIdx.y) * unroll_size; i < size;
         i += blockDim.y * gridDim.y * unroll_size)
    {
        float yi[unroll_size] = {};
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < size; j += blockDim.x * gridDim.x)
        {
#pragma unroll
            for (int k = 0; k < unroll_size; k++)
            {
                yi[k] += a[j + k * size] * vector[j];
            }
        }
        for (int k = 0; k < unroll_size; k++)
        {
            atomicAdd(&result[i + k], yi[k]);
        }
        a += blockDim.y * gridDim.y * unroll_size * size;
    }
}