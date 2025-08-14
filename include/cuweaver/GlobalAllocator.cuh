/**
* @file GlobalAllocator.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Defines a CUDA-aware allocator for managing global memory via `cudaMalloc`/`cudaFree`.
 *
 * @details Implements the C++ allocator concept for CUDA global memory. Provides type-safe allocation,
 *          deallocation, and memory copy operations. Uses `CUW_THROW_IF_ERROR` to propagate CUDA errors
 *          and falls back to standard limits if memory info retrieval fails.
 */
#ifndef CUWEAVER_GLOBALALLOCATOR_CUH
#define CUWEAVER_GLOBALALLOCATOR_CUH

#ifdef __CUDACC__

#include <limits>
#include <cuweaver_utils/ErrorCheck.cuh>

namespace cuweaver {
    /**
     * @class GlobalAllocator<T> GlobalAllocator.cuh
     * @brief CUDA global memory allocator using `cudaMalloc` and `cudaFree`.
     *
     * @details Implements type-safe allocation/deallocation for CUDA global memory. Complies with the
     *          C++ allocator concept and includes CUDA-specific memory copy functionality.
     *
     * @tparam T Type of elements to allocate in global memory.
     */
    template <typename T>
    class GlobalAllocator {
    public:
        using valueType = T; //!< Element type allocated by the allocator.
        using pointer = T*; //!< Pointer to allocated elements.
        using constPointer = const T*; //!< Const pointer to allocated elements.
        using voidPointer = void*; //!< Generic void pointer type.
        using constVoidPointer = const void*; //!< Const generic void pointer type.
        using differenceType = ptrdiff_t; //!< Type for pointer arithmetic differences.
        using sizeType = size_t; //!< Type for memory block sizes (number of elements).

        /**
         * @brief Rebind template to get an allocator for a different element type.
         *
         * @details Converts a `GlobalAllocator<T>` to a `GlobalAllocator<U>` for type-safe memory operations.
         * @tparam U Target element type for the rebound allocator.
         */
        template <typename U>
        struct rebind {
            using other = GlobalAllocator<U>;
        };

        /**
         * @brief Default constructor.
         *
         * @details Creates a stateless allocator instance (no memory allocated during construction).
         */
        GlobalAllocator() noexcept = default;

        /**
         * @brief Copy constructor from another `GlobalAllocator` instance.
         *
         * @details Trivial copy constructor (allocator has no state to copy).
         * @tparam U Element type of the source allocator.
         * @param[in] Other source `GlobalAllocator` (unused, as allocator is stateless).
         */
        template <typename U>
        GlobalAllocator(const GlobalAllocator<U>&) noexcept {
        }

        /**
         * @brief Allocates global memory for `n` elements of type `T`.
         *
         * @details Uses `cudaMalloc` to allocate `n * sizeof(T)` bytes. Throws `std::bad_alloc` if:
         *          - `n` exceeds `maxSize()` (insufficient free memory)
         *          - `cudaMalloc` returns an error
         *          Returns `nullptr` if `n` is 0.
         *
         * @param[in] n Number of elements to allocate (not bytes).
         * @return Pointer to allocated global memory, or `nullptr` if `n == 0`.
         * @throws std::bad_alloc If allocation fails or `n` is too large.
         */
        pointer allocate(sizeType n) {
            if (n == 0) {
                return nullptr;
            }
            if (n > maxSize()) {
                throw std::bad_alloc();
            }
            pointer ptr = nullptr;
            CUW_THROW_IF_ERROR(cudaMalloc(&ptr, n*sizeof(T)));
            return ptr;
        }

        /**
         * @brief Frees global memory previously allocated by `allocate`.
         *
         * @details Uses `cudaFree` to release memory. The `n` parameter is unused (required for allocator compliance).
         * @param[in] p Pointer to memory block to free (must be from `allocate`).
         * @param[in] n Number of elements in the block (unused).
         * @note Does nothing if `p` is `nullptr`.
         */
        void deallocate(pointer p, sizeType n) noexcept {
            if (p != nullptr) {
                cudaFree(p);
            }
            p = nullptr;
        }

        /**
         * @brief Copies `n` elements using `cudaMemcpy`.
         *
         * @details Uses CUDA's `cudaMemcpy` to copy data between memory regions. The copy direction is
         *          specified by `kind`. Throws if `cudaMemcpy` fails.
         *
         * @tparam U Type of source/destination pointers (must be compatible with `T`).
         * @param[in,out] dst Destination memory pointer.
         * @param[in] src Source memory pointer.
         * @param[in] n Number of elements to copy (not bytes).
         * @param[in] kind CUDA copy direction (e.g., `cudaMemcpyHostToDevice`).
         * @throws std::runtime_error If `cudaMemcpy` returns an error.
         */
        template <typename U>
        void memcpy(U* dst, U* src, sizeType n, cudaMemcpyKind kind) {
            CUW_THROW_IF_ERROR(cudaMemcpy(dst, src, n * sizeof(T), kind));
        }

        /**
         * @brief Returns the maximum number of elements that can be allocated.
         *
         * @details Tries to get free global memory via `cudaMemGetInfo`. If that fails, falls back to
         *          `std::numeric_limits<sizeType>::max() / sizeof(T)`.
         * @return Maximum number of `T` elements that can be allocated in a single call.
         */
        sizeType maxSize() const noexcept {
            sizeType freeMem, totalMem;
            if (const auto err = cudaMemGetInfo(&freeMem, &totalMem); err != cudaSuccess) {
                return std::numeric_limits<sizeType>::max() / sizeof(T);
            }
            return freeMem / sizeof(T);
        }
    };
}

#endif

#ifndef __CUDACC__
#pragma warning("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_GLOBALALLOCATOR_CUH
