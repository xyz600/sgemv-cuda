#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace cuda
{
    static void check_error(const ::cudaError_t e, const char* f, decltype(__LINE__) n)
    {
        if (e != ::cudaSuccess)
        {
            std::stringstream s;
            s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": " << ::cudaGetErrorString(e);
            throw std::runtime_error{s.str()};
        }
    }
}  // namespace cuda

#define CHECK_CUDA_ERROR(e) (cuda::check_error(e, __FILE__, __LINE__))

namespace cuda
{
    struct deleter
    {
        void operator()(void* p) const { CHECK_CUDA_ERROR(::cudaFree(p)); }
    };

    template <typename T>
    using unique_ptr = std::unique_ptr<T, deleter>;

    // auto array = cuda::make_unique<float[]>(n);
    // ::cudaMemcpy(array.get(), src_array, sizeof(float)*n, ::cudaMemcpyHostToDevice);
    template <typename T>
    typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(const std::size_t n)
    {
        using U = typename std::remove_extent<T>::type;
        U* p;
        CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
        return cuda::unique_ptr<T>{p};
    }

    // auto value = cuda::make_unique<my_class>();
    // ::cudaMemcpy(value.get(), src_value, sizeof(my_class), ::cudaMemcpyHostToDevice);
    template <typename T>
    cuda::unique_ptr<T> make_unique()
    {
        T* p;
        CHECK_CUDA_ERROR(::cudaMallocManaged(reinterpret_cast<void**>(&p), sizeof(T)));
        return cuda::unique_ptr<T>{p};
    }

    // auto array = cuda::make_unique_managed<float[]>(n);
    // ::cudaMemcpy(array.get(), src_array, sizeof(float)*n, ::cudaMemcpyHostToDevice);
    template <typename T>
    typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique_managed(const std::size_t n)
    {
        using U = typename std::remove_extent<T>::type;
        U* p;
        CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
        return cuda::unique_ptr<T>{p};
    }

    // auto value = cuda::make_unique_managed<my_class>();
    // ::cudaMemcpy(value.get(), src_value, sizeof(my_class), ::cudaMemcpyHostToDevice);
    template <typename T>
    cuda::unique_ptr<T> make_unique_managed()
    {
        T* p;
        CHECK_CUDA_ERROR(::cudaMallocManaged(reinterpret_cast<void**>(&p), sizeof(T)));
        return cuda::unique_ptr<T>{p};
    }

}  // namespace cuda