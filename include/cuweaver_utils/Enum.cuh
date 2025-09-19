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

    /**
     * @enum cudaDeviceFlags
     * @brief Enumerates CUDA device scheduling policies and host memory mapping combinations.
     *
     * @details Defines strategies for CPU thread behavior while waiting for CUDA device operations,
     *          along with options to enable **device-accessible pinned host memory** (via `MapHost` suffix).
     *          Scheduling modes can be combined with memory mapping using bitwise OR (e.g., `AutoMapHost = Auto | MapHost`).
     */
    enum class cudaDeviceFlags {
        Auto = 0x00, //!< Default heuristic: spins if active CUDA contexts ≤ logical processors, else yields. Tegra devices may use BlockingSync for low power.
        Spin = 0x01, //!< Active spinning during device waits: reduces latency but may impact parallel CPU threads.
        Yield = 0x02, //!< Yields CPU thread during waits: increases latency but improves parallel CPU thread performance.
        BlockingSync = 0x04, //!< Blocks CPU thread on a synchronization primitive during device waits.
        AutoMapHost = 0x08, //!< Auto scheduling + enable device-accessible pinned host memory allocation.
        SpinMapHost = 0x09, //!< Spin scheduling + enable device-accessible pinned host memory allocation.
        YieldMapHost = 0x0A, //!< Yield scheduling + enable device-accessible pinned host memory allocation.
        BlockingSyncMapHost = 0x0B //!< BlockingSync scheduling + enable device-accessible pinned host memory allocation.
    };

    /**
     * @enum cudaMemMetaFlags
     * @brief Flags indicating the intended access direction for encapsulated CUDA memory.
     *
     * @details Used to mark whether the next operation on the encapsulated CUDA memory
     *          will be a write or a read. This helps manage memory access semantics
     *          during CUDA operations.
     */
    enum class cudaMemMetaFlags {
        WRITE, //!< Marks the memory for an upcoming write operation.
        READ   //!< Marks the memory for an upcoming read operation.
    };

    /**
     * @namespace deviceFlags
     * @brief Namespace containing device selection flags for the library's stream manager.
     *
     * @details Encapsulates enumeration flags used by the library's stream manager to determine
     *          which device a CUDA operation should execute on.
     */
    namespace deviceFlags {
        /**
         * @enum deviceFlags::type
         * @brief Device selection flags for the library's stream manager when scheduling CUDA operations.
         *
         * @details Determines the target device for a CUDA operation during scheduling:
         *          - Auto: Automatically selects the device with the largest amount of memory among
         *            those associated with the memory involved in the operation.
         *          - Current: Continues using the currently active device without switching.
         *          - Host: Marks host memory, typically used for operations (e.g., memcpy) involving host memory.
         */
        enum type {
            Auto = 0xfffffff, //!< Automatically select device with the most memory from involved devices.
            Current = 0xffffffe, //!< Use current device without switching.
            Host = 0xffffffd //!< Mark host memory (for operations like memcpy involving host memory).
        };
    }

    /**
     * @enum memcpyFlags
     * @brief Flags specifying the direction of a memory copy operation.
     *
     * @details Used to define the source and target locations for CUDA memory copy operations.
     *          Each flag represents a distinct combination of source (host or device)
     *          and target (host or device), or a default behavior.
     */
    enum class memcpyFlags {
        HostToHost = 0, //!< Memory copy from host to host.
        HostToDevice = 1, //!< Memory copy from host to CUDA device.
        DeviceToHost = 2, //!< Memory copy from CUDA device to host.
        DeviceToDevice = 3, //!< Memory copy from CUDA device to another CUDA device.
        Default = 4 //!< Use default copy direction or automatic detection.
    };
}

#endif //CUWEAVER_ENUM_CUH
