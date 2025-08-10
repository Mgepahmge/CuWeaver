/**
 * @file Enum.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Enumeration definitions for the CuWeaver library.
 */
#ifndef CUWEAVER_ENUM_CUH
#define CUWEAVER_ENUM_CUH

namespace cuweaver {
    /**
     * @enum cudaEventFlags
     * @brief Type-safe flags for configuring CUDA event behavior.
     *
     * @details Maps to CUDA's native `cudaEventFlags` values, controlling synchronization, timing, and inter-process capabilities of the event.
     */
    enum class cudaEventFlags {
        Default = 0x00,
        BlockingSync = 0x01,
        DisableTiming = 0x02,
        Interprocess = 0x04,
    };
}

#endif //CUWEAVER_ENUM_CUH