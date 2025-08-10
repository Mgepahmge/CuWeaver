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

    /**
     * @enum cudaStreamFlags
     * @brief Type-safe enumeration of CUDA stream configuration flags.
     *
     * @details Defines standard bitmask flags for configuring CUDA stream behavior. Values align with
     * CUDA's native stream flag constants and enable type-safe initialization of `cudaStream` instances.
     */
    enum class cudaStreamFlags {
        Default = 0x00,
        NonBlocking = 0x01,
    };

    /**
     * @enum cuweaver::cudaEventRecordFlags
     * @brief Type-safe flags for configuring CUDA event recording behavior.
     *
     * @details Maps to CUDA's native flags for `cudaEventRecordWithFlags`, controlling how an event is scheduled
     *          into a CUDA stream. These flags adjust the event's compatibility with external systems
     *          or synchronization requirements.
     */
    enum class cudaEventRecordFlags {
        Default = 0x00,
        External = 0x01,
    };

    /**
    * @enum cudaEventWait
    * @brief Flags defining CUDA event wait behavior options.
    *
    * @details Specifies configuration options for operations that wait on CUDA events (e.g., stream waits for an event).
    */
    enum class cudaEventWait {
        Default = 0x00,
        External = 0x01,
    };
}

#endif //CUWEAVER_ENUM_CUH
