/**
 * @file cudaMemMeta.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Defines CUDA memory metadata utilities for CuWeaver.
 *
 * @details This file declares the `detail::cudaMemMeta` class for encapsulating CUDA memory
 *          metadata (pointer, device ID, and access flags), along with type traits and helper
 *          functions to create metadata instances. These utilities are only available when
 *          compiling with a CUDA compiler (__CUDACC__ defined).
 */
#ifndef CUWEAVER_CUDAMEMMETA_CUH
#define CUWEAVER_CUDAMEMMETA_CUH

#ifdef __CUDACC__
#include <type_traits>
#include <cuweaver_utils/Enum.cuh>

namespace cuweaver {
    namespace detail {
        /**
         * @class cudaMemMeta<T> cudaMemMeta.cuh
         * @brief Encapsulates metadata for a CUDA memory pointer.
         *
         * @details Stores a CUDA memory pointer along with its associated device ID and access flags.
         *          Enforces that the template parameter `T` is a pointer type via a static assertion.
         *
         * @tparam T Pointer type to the CUDA memory (e.g., `float*`, `int*`).
         */
        template <typename T>
        class cudaMemMeta {
            static_assert(std::is_pointer_v<T>, "T must be a pointer type");

        public:
            /**
             * @brief Constructs a cudaMemMeta instance with specified memory, device ID, and flags.
             *
             * @param[in] data Pointer to the CUDA memory.
             * @param[in] deviceId CUDA device ID associated with the memory.
             * @param[in] flags Metadata flags (e.g., read/write permissions).
             */
            cudaMemMeta(T data, const int deviceId, const cudaMemMetaFlags flags) : data(data), deviceId(deviceId),
                flags(flags) {
            }

            T data; //!< Pointer to the CUDA memory.
            int deviceId; //!< CUDA device ID where the memory resides.
            cudaMemMetaFlags flags; //!< Flags describing memory access permissions.
        };
    }

    /**
     * @struct isCudaMemMeta<T>
     * @brief Type trait to check if a type is a `detail::cudaMemMeta` specialization.
     *
     * @tparam T Type to check.
     */
    template <typename T>
    struct isCudaMemMeta : std::false_type {};

    /**
     * @struct isCudaMemMeta<detail::cudaMemMeta<T>>
     * @brief Specialization of `isCudaMemMeta` for `detail::cudaMemMeta` types (returns true).
     *
     * @tparam T Pointer type of the `detail::cudaMemMeta` specialization.
     */
    template <typename T>
    struct isCudaMemMeta<detail::cudaMemMeta<T>> : std::true_type {};

    /**
     * @var isCudaMemMeta_v<T>
     * @brief Constexpr alias for `isCudaMemMeta<T>::value`.
     *
     * @tparam T Type to check.
     */
    template <typename T>
    constexpr bool isCudaMemMeta_v = isCudaMemMeta<T>::value;

    /**
     * @brief Creates a `detail::cudaMemMeta` instance with WRITE access flag.
     *
     * @tparam T Pointer type to the CUDA memory.
     * @param[in] data Pointer to the CUDA memory.
     * @param[in] deviceID CUDA device ID associated with the memory.
     * @return `detail::cudaMemMeta<T>` instance with WRITE flag.
     */
    template <typename T>
    auto makeWrite(T data, const int deviceID) {
        return detail::cudaMemMeta<T>(data, deviceID, cudaMemMetaFlags::WRITE);
    }

    /**
     * @brief Creates a `detail::cudaMemMeta` instance with READ access flag.
     *
     * @tparam T Pointer type to the CUDA memory.
     * @param[in] data Pointer to the CUDA memory.
     * @param[in] deviceID CUDA device ID associated with the memory.
     * @return `detail::cudaMemMeta<T>` instance with READ flag.
     */
    template <typename T>
    auto makeRead(T data, const int deviceID) {
        return detail::cudaMemMeta<T>(data, deviceID, cudaMemMetaFlags::READ);
    }
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_CUDAMEMMETA_CUH
