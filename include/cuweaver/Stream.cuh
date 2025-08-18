/**
 * @file Stream.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Declares the cuweaver::cudaStream class for RAII management of CUDA streams.
 *
 * @details This header file defines the `cuweaver::cudaStream` class, which provides a RAII (Resource Acquisition Is Initialization) wrapper for CUDA stream handles (`cudaStream_t`). It includes type aliases for stream configuration flags (`cudaStreamFlags_t`) and priorities (`cudaStreamPriority_t`), a type-safe enumeration (`cudaStreamFlags`) for standard stream flags, static methods to retrieve CUDA stream priority bounds, and member function declarations for stream lifecycle and state management.
 *
 * The header uses conditional compilation (`#ifdef __CUDACC__`) to ensure the `cudaStream` class is only declared when the CUDA compiler is active. If CUDA is not available (i.e., `__CUDACC__` is undefined), a warning is emitted indicating the file will not be compiled.
 */
#ifndef CUWEAVER_STREAM_CUH
#define CUWEAVER_STREAM_CUH

#ifdef __CUDACC__
#include <cuweaver_utils/Enum.cuh>

namespace cuweaver {
    /**
     * @class cuweaver::cudaStream [cuweaver_stream.cuh]
     * @brief RAII wrapper for CUDA streams to manage lifecycle and configuration.
     *
     * @details Provides a type-safe, RAII-compliant interface for CUDA stream handles (`cudaStream_t`),
     * automatically creating and destroying streams to avoid resource leaks. Supports move semantics
     * (copy operations are deleted to prevent double-free), type-safe configuration flags via the
     * `cudaStreamFlags` enumeration, and static methods to retrieve CUDA stream priority bounds.
     * Tracks the underlying stream handle, configuration flags, and priority for inspection.
     *
     * Type aliases `cudaStreamFlags_t` (raw unsigned integer for stream flags), `cudaStreamPriority_t`
     * (integer for stream priority values) and `cudaStreamId_t` (unsigned long long for stream IDs) clarify parameter types. The `cudaStreamFlags` enum defines
     * standard stream options: `Default` (0x00) for default behavior and `NonBlocking` (0x01) for
     * non-blocking stream synchronization. A static `DefaultPriority` constant sets the default stream
     * priority to 0.
     */
    class cudaStream {
    public:
        using cudaStreamFlags_t = unsigned int;
        using cudaStreamPriority_t = int;
        using cudaStreamId_t = unsigned long long;

        constexpr static cudaStreamPriority_t DefaultPriority = 0;

        /**
         * @brief Gets the least (highest numerical value) valid CUDA stream priority for the current device.
         *
         * @return The minimum valid priority value, obtained via `cudaDeviceGetStreamPriorityRange`.
         */
        static int getLeastPriority() noexcept;

        /**
         * @brief Gets the greatest (lowest numerical value) valid CUDA stream priority for the current device.
         *
         * @return The maximum valid priority value, obtained via `cudaDeviceGetStreamPriorityRange`.
         */
        static int getGreatestPriority() noexcept;

        /**
         * @brief Checks if a given CUDA stream priority is within the valid range.
         *
         * @param[in] priority The priority value to validate.
         * @return True if `priority` is between `getGreatestPriority()` and `getLeastPriority()` (inclusive); false otherwise.
         */
        static bool isPriorityValid(cudaStreamPriority_t priority) noexcept;

        /**
         * @brief Gets the CUDA default stream (null stream).
         *
         * @details Returns a `cudaStream` instance representing the CUDA default stream (also known as the null stream).
         * The default stream executes work sequentially and synchronizes with all other streams in the same CUDA context.
         * This function is marked `noexcept` and does not throw exceptions, as it constructs the stream from a null handle
         * (no CUDA API calls are made that could fail).
         *
         * @par Parameters
         *      None.
         *
         * @return `cudaStream` instance wrapping the CUDA default stream (null handle).
         */
        static cudaStream defaultStream() noexcept;

        /**
         * @brief Constructs a CUDA stream with default flags and priority.
         *
         * @details Initializes the stream using `cudaStreamFlags::Default` (0x00) and `DefaultPriority` (0),
         * then creates the stream via `cudaStreamCreateWithPriority`. If `cudaStreamCreateWithPriority` fails
         * (returns a non-cudaSuccess status), a `cuweaver::cudaError` is thrown with contextual error information.
         *
         * @throws cuweaver::cudaError Thrown if `cudaStreamCreateWithPriority` fails to create the CUDA stream.
         */
        cudaStream();

        /**
         * @brief Constructs a CUDA stream with specified type-safe flags and priority.
         *
         * @param[in] flags Configuration flags from the `cudaStreamFlags` enumeration.
         * @param[in] priority Priority of the stream (defaults to `DefaultPriority`).
         *
         * @details Validates `priority` against the range from `getGreatestPriority()` to `getLeastPriority()`.
         * Uses `DefaultPriority` if the input is invalid. Converts the enum flags to the native `cudaStreamFlags_t` type,
         * then creates the stream via `cudaStreamCreateWithPriority` with the provided flags and (corrected) priority.
         * If `cudaStreamCreateWithPriority` fails (returns a non-cudaSuccess status), a `cuweaver::cudaError` is thrown
         * with contextual error information.
         *
         * @throws cuweaver::cudaError Thrown if `cudaStreamCreateWithPriority` fails to create the CUDA stream.
         */
        explicit cudaStream(cudaStreamFlags flags, cudaStreamPriority_t priority = DefaultPriority);

        /**
         * @brief Constructs a CUDA stream with raw flags and specified priority.
         *
         * @param[in] flags Raw unsigned integer representing stream configuration flags.
         * @param[in] priority Priority of the stream (defaults to `DefaultPriority`).
         *
         * @details Similar to the `cudaStreamFlags` overload, but accepts a raw `cudaStreamFlags_t` value for flags.
         * Validates `priority` against the range from `getGreatestPriority()` to `getLeastPriority()`; uses
         * `DefaultPriority` if the input is invalid. Creates the stream via `cudaStreamCreateWithPriority` with
         * the provided raw flags and (corrected) priority. If `cudaStreamCreateWithPriority` fails (returns a
         * non-cudaSuccess status), a `cuweaver::cudaError` is thrown with contextual error information.
         *
         * @throws cuweaver::cudaError Thrown if `cudaStreamCreateWithPriority` fails to create the CUDA stream.
         */
        explicit cudaStream(cudaStreamFlags_t flags, cudaStreamPriority_t priority = DefaultPriority);

        /**
         * @brief Constructs a `cudaStream` wrapper around an existing CUDA stream handle and initializes its configuration.
         *
         * @details Takes ownership of the provided `cudaStream_t` handle. Queries the stream's unique identifier, priority,
         *          and flags using `cudaStreamGetId`, `cudaStreamGetPriority`, and `cudaStreamGetFlags` respectively,
         *          and initializes the corresponding `id`, `priority`, and `flags` member variables with the retrieved values.
         *
         * @param[in] stream Existing CUDA stream handle to manage.
         *
         * @throws cuweaver::cudaError Thrown if any of the underlying `cudaStreamGetId`, `cudaStreamGetPriority`,
         *                             or `cudaStreamGetFlags` calls fail (e.g., invalid stream handle or CUDA runtime error).
         */
        explicit cudaStream(cudaStream_t stream);

        cudaStream(const cudaStream&) = delete;

        cudaStream& operator=(const cudaStream&) = delete;

        /**
         * @brief Moves ownership of a CUDA stream from another `cudaStream` instance.
         *
         * @details Transfers the underlying stream handle, flags, and priority from `other` to this instance.
         * Sets `other`'s stream handle to `nullptr` to prevent double destruction.
         *
         * @param[in,out] other The `cudaStream` instance to move resources from.
         */
        cudaStream(cudaStream&& other) noexcept;

        /**
         * @brief Moves ownership of a CUDA stream to this instance via assignment.
         *
         * @details Handles self-assignment by checking identity. Resets this instance's stream to the handle
         * from `other`, transfers flags/priority, and nulls `other`'s stream handle to avoid resource leaks.
         *
         * @param[in,out] other The `cudaStream` instance to move resources from.
         * @return A reference to this `cudaStream` instance after assignment.
         */
        cudaStream& operator=(cudaStream&& other) noexcept;

        /**
         * @brief Destroys the CUDA stream and releases associated resources.
         *
         * @details Automatically destroys the underlying `cudaStream_t` handle if the stream is valid
         * (checked via `isValid()`).
         */
        ~cudaStream();

        /**
         * @brief Retrieves the raw CUDA stream handle managed by this instance.
         *
         * @return The underlying `cudaStream_t` handle (valid only if `isValid()` returns true).
         */
        [[nodiscard]] cudaStream_t nativeHandle() const noexcept;

        /**
         * @brief Gets the configuration flags used to create the CUDA stream.
         *
         * @return The stream's configuration flags as a `cudaStreamFlags_t` value.
         */
        [[nodiscard]] cudaStreamFlags_t getFlags() const noexcept;

        /**
         * @brief Gets the priority value used to create the CUDA stream.
         *
         * @return The stream's priority as a `cudaStreamPriority_t` value.
         */
        [[nodiscard]] cudaStreamPriority_t getPriority() const noexcept;

        /**
         * @brief Gets the unique identifier of the CUDA stream.
         *
         * @details Returns the internal unique identifier associated with this CUDA stream instance. This method
         *          is `const` (it does not modify the stream object) and `noexcept` (it will never throw exceptions).
         *
         * @return The unique identifier of the stream, of type `cudaStream::cudaStreamId_t`.
         */
        [[nodiscard]] cudaStreamId_t getId() const noexcept;

        /**
         * @brief Resets the `cudaStream` wrapper with a new CUDA stream handle and updates its configuration attributes.
         *
         * @details First destroys the currently managed CUDA stream (if valid, via `resetStream()`), takes ownership of
         *          the provided `stream` handle, then synchronizes the wrapper's `id`, `priority`, and `flags` members
         *          with the new stream's actual configuration by calling `updateAttributes()`. If `stream` is `nullptr`,
         *          the instance becomes invalid until a valid handle is assigned.
         *
         * @param[in] stream New CUDA stream handle to manage (defaults to `nullptr`). If `nullptr`, subsequent operations
         *                   on this instance may fail until a valid handle is provided.
         *
         * @throws cuweaver::cudaError Thrown if `updateAttributes()` fails to query the new stream's attributes (e.g.,
         *                             invalid stream handle or CUDA runtime error during `cudaStreamGetId`,
         *                             `cudaStreamGetPriority`, or `cudaStreamGetFlags` calls).
         *
         * @par Returns
         *      Nothing.
         */
        void reset(cudaStream_t stream = nullptr);

        /**
         * @brief Checks if the instance manages a valid CUDA stream handle.
         *
         * @return True if the underlying `cudaStream_t` handle is non-null; false otherwise.
         */
        [[nodiscard]] bool isValid() const noexcept;

    private:
        /**
         * @brief Updates the stream's attributes (ID, priority, flags) from the underlying CUDA stream.
         *
         * @details Synchronizes the wrapper class's `id`, `priority`, and `flags` member variables with the actual
         *          attributes of the managed CUDA stream. This is done by invoking `cudaStreamGetId` to retrieve the stream's
         *          unique identifier, `cudaStreamGetPriority` for the stream's priority value, and `cudaStreamGetFlags` for
         *          the stream's configuration flags. This ensures the wrapper's state matches the current state of the
         *          underlying stream.
         *
         * @par Parameters
         *      None.
         *
         * @throws cuweaver::cudaError Thrown if any of the underlying CUDA API calls (`cudaStreamGetId`,
         *                             `cudaStreamGetPriority`, or `cudaStreamGetFlags`) fails (e.g., invalid stream handle
         *                             or CUDA runtime error).
         *
         * @par Returns
         *      Nothing.
         */
        void updateAttributes();

        /**
         * @brief Resets the `cudaStream` wrapper with a new CUDA stream handle.
         *
         * @details Destroys the currently managed CUDA stream handle (if valid, as determined by `isValid()`)
         *          using `cudaStreamDestroy`, then takes ownership of the provided new `cudaStream_t` handle.
         *          This method is `noexcept` and will never throw exceptions.
         *
         * @param[in] stream New CUDA stream handle to manage. If `nullptr` or an invalid handle is provided,
         *                   subsequent operations on this `cudaStream` instance may fail.
         *
         * @par Returns
         *      Nothing.
         */
        void resetStream(cudaStream_t stream) noexcept;

        cudaStream_t stream; //!< Underlying CUDA stream handle managed by this wrapper.
        cudaStreamFlags_t flags; //!< Configuration flags for the CUDA stream (raw unsigned integer).
        cudaStreamPriority_t priority; //!< Priority value assigned to the CUDA stream.
        cudaStreamId_t id; //!< Unique identifier for the CUDA stream.
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_STREAM_CUH
