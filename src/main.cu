#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>

#include "cuda_utility.cuh"

__global__ void sgemv_dev(const float *__restrict__ matrix, const float *__restrict__ vector, float *result,
                          const int size);

void initialize_matrix(float *matrix, float *vector, int size, float init)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i * size + j] = init * static_cast<float>((((i + 1) % 4 + j % 4) % 8)) / 6.0f;
        }
        init = init / 2.8f + 0.87f;
        vector[i] = init * static_cast<float>((i % 16 + 1)) / 16.0f;
        init = init / 2.8f + 1.23f;
    }
}

void sgemv_host(const float *matrix, const float *vector, float *result, const std::size_t size)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < size; i++)
    {
        for (std::size_t j = 0; j < size; j++)
        {
            result[i] += matrix[i * size + j] * vector[j];
        }
    }
}

int main()
{
    constexpr std::size_t matrix_size = 32768;
    constexpr std::size_t max_iter = 5;

    const auto matrix_dev = cuda::make_unique<float[]>(matrix_size * matrix_size);
    const auto vector_dev = cuda::make_unique<float[]>(matrix_size);
    const auto answer_dev = cuda::make_unique<float[]>(matrix_size);

    const auto matrix_host = std::make_unique<float[]>(matrix_size * matrix_size);
    const auto vector_host = std::make_unique<float[]>(matrix_size);
    const auto answer_host = std::make_unique<float[]>(matrix_size);

    initialize_matrix(matrix_host.get(), vector_host.get(), matrix_size, 0.3);
    for (std::size_t i = 0; i < matrix_size; i++)
    {
        answer_host[i] = 0.0;
    }

    CHECK_CUDA_ERROR(::cudaMemcpy(matrix_dev.get(), matrix_host.get(), sizeof(float) * matrix_size * matrix_size,
                                  cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        ::cudaMemcpy(vector_dev.get(), vector_host.get(), sizeof(float) * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        ::cudaMemcpy(answer_dev.get(), answer_host.get(), sizeof(float) * matrix_size, cudaMemcpyHostToDevice));

    constexpr double datasize_GB = max_iter * sizeof(float) * matrix_size * matrix_size / (1024.0 * 1024.0 * 1024.0);

    {
        const auto start = std::chrono::system_clock::now();
        for (std::size_t iter = 0; iter < max_iter; iter++)
        {
            sgemv_host(matrix_host.get(), vector_host.get(), answer_host.get(), matrix_size);
        }
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        std::cout << "elapsed: " << elapsed << "[ms]" << std::endl;
        std::cout << "throughput: " << (datasize_GB / (elapsed / 1000.0)) << "[GB/s]" << std::endl;
    }

    {
        dim3 grid(8, 7);
        dim3 block(32, 8);

        {
            // warm up
            const auto answer_dev_tmp = cuda::make_unique<float[]>(matrix_size);
            sgemv_dev<<<grid, block>>>(matrix_dev.get(), vector_dev.get(), answer_dev_tmp.get(), matrix_size);
        }

        cudaEvent_t start, stop;

        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        for (std::size_t iter = 0; iter < max_iter; iter++)
        {
            sgemv_dev<<<grid, block>>>(matrix_dev.get(), vector_dev.get(), answer_dev.get(), matrix_size);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

        float elapsed = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));

        std::cout << "elapsed: " << elapsed << "[ms]" << std::endl;
        std::cout << "throughput: " << (datasize_GB / (elapsed / 1000.0)) << "[GB/s]" << std::endl;
    }

    {
        auto answer_dev_host = std::make_unique<float[]>(matrix_size);
        CHECK_CUDA_ERROR(
            cudaMemcpy(answer_dev_host.get(), answer_dev.get(), sizeof(float) * matrix_size, cudaMemcpyDeviceToHost));
        float ans = 0;
        for (std::size_t i = 0; i < matrix_size; i++)
        {
            ans += std::abs(answer_host[i] - answer_dev_host[i]) / std::min(answer_host[i], answer_dev_host[i]);
        }
        std::cout << "diff: " << (ans / matrix_size) << std::endl;
    }

    return 0;
}