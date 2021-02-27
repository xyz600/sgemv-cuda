#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>

#include "cuda_utility.cuh"
#include "sgemv_kernel.cuh"

__global__ void sgemv_dev(const int m, const int n, const float *__restrict__ A, const float *__restrict__ x, float *y);

void initialize_matrix(float *matrix, float *vector, int row_size, int column_size, float init)
{
#pragma omp parallel for
    for (int i = 0; i < row_size; i++)
    {
        for (int j = 0; j < column_size; j++)
        {
            matrix[i * column_size + j] = init * static_cast<float>((((i + 1) % 4 + j % 4) % 8)) / 6.0f;
        }
        init = init / 2.8f + 0.87f;
    }
    for (int j = 0; j < column_size; j++)
    {
        vector[j] = init * static_cast<float>((j % 16 + 1)) / 16.0f;
        init = init / 2.8f + 1.23f;
    }
}

void sgemv_host(const std::size_t n, const std::size_t m, const float *matrix, const float *vector, float *result)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < m; j++)
        {
            result[i] += matrix[i * m + j] * vector[j];
        }
    }
}

int main()
{
    constexpr std::size_t row_size = 32768;
    constexpr std::size_t column_size = row_size;
    constexpr std::size_t max_iter = 5;

    const auto matrix_dev = cuda::make_unique<float[]>(row_size * column_size);
    const auto vector_dev = cuda::make_unique<float[]>(column_size);
    const auto answer_dev = cuda::make_unique<float[]>(row_size);

    const auto matrix_host = std::make_unique<float[]>(row_size * column_size);
    const auto vector_host = std::make_unique<float[]>(column_size);
    const auto answer_host = std::make_unique<float[]>(row_size);

    initialize_matrix(matrix_host.get(), vector_host.get(), row_size, column_size, 0.3);
    for (std::size_t i = 0; i < row_size; i++)
    {
        answer_host[i] = 0.0;
    }

    CHECK_CUDA_ERROR(::cudaMemcpy(matrix_dev.get(), matrix_host.get(), sizeof(float) * row_size * column_size,
                                  cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        ::cudaMemcpy(vector_dev.get(), vector_host.get(), sizeof(float) * column_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        ::cudaMemcpy(answer_dev.get(), answer_host.get(), sizeof(float) * row_size, cudaMemcpyHostToDevice));

    constexpr double datasize_GB = max_iter * sizeof(float) * row_size * column_size / (1024.0 * 1024.0 * 1024.0);

    {
        const auto start = std::chrono::system_clock::now();
        for (std::size_t iter = 0; iter < max_iter; iter++)
        {
            sgemv_host(row_size, column_size, matrix_host.get(), vector_host.get(), answer_host.get());
        }
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        std::cout << "elapsed: " << elapsed << "[ms]" << std::endl;
        std::cout << "throughput: " << (datasize_GB / (elapsed / 1000.0)) << "[GB/s]" << std::endl;
    }

    {
        dim3 grid(2, (row_size + block_size - 1) / block_size);
        dim3 block(block_size, div_size);

        {
            // warm up
            const auto answer_dev_tmp = cuda::make_unique<float[]>(row_size);
            sgemv_dev<<<grid, block>>>(row_size, column_size, matrix_dev.get(), vector_dev.get(), answer_dev_tmp.get());
        }

        cudaEvent_t start, stop;

        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        for (std::size_t iter = 0; iter < max_iter; iter++)
        {
            sgemv_dev<<<grid, block>>>(row_size, column_size, matrix_dev.get(), vector_dev.get(), answer_dev.get());
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
        auto answer_dev_host = std::make_unique<float[]>(row_size);
        CHECK_CUDA_ERROR(
            cudaMemcpy(answer_dev_host.get(), answer_dev.get(), sizeof(float) * row_size, cudaMemcpyDeviceToHost));
        float ans = 0;

        for (std::size_t i = 0; i < row_size; i++)
        {
            ans = std::max(
                ans, std::abs(answer_host[i] - answer_dev_host[i]) / std::min(answer_host[i], answer_dev_host[i]));
        }
        std::cout << "diff: " << ans << std::endl;
    }

    return 0;
}