/**
 * @file Event.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief RAII wrapper for CUDA events.
 *
 * @details Provides a type-safe, exception-safe wrapper around CUDA events (`cudaEvent_t`) using RAII (Resource Acquisition Is Initialization) semantics.
 * Manages the lifecycle of CUDA events, including creation and destruction, and offers type-safe flags for event configuration.
 */
#ifndef CUWEAVER_EVENT_CUH
#define CUWEAVER_EVENT_CUH

#ifdef __CUDACC__
#include <cuweaver_utils/Enum.cuh>

namespace cuweaver {
    /**
     * @class cuweaver::cudaEvent
     * @brief RAII wrapper for CUDA events.
     *
     * @details Manages the lifecycle of a CUDA event (`cudaEvent_t`) using RAII (Resource Acquisition Is Initialization) semantics. Automatically creates the event on construction and destroys it on destruction to prevent resource leaks. Supports move semantics for safe ownership transfer, while copy semantics are disabled to avoid duplicate management of the same event. Provides a type-safe enumeration (`cudaEventFlags`) for configuring event behavior, aligned with CUDA's native event flag values.
     */
    class cudaEvent {
    public:
        using cudaEventFlags_t = unsigned int;

        /**
         * @brief Constructs a CUDA event with default flags.
         *
         * @details Initializes the underlying CUDA event handle to `nullptr`, sets flags to `cudaEventFlags::Default`,
         * then creates the event via `cudaEventCreateWithFlags` using default configuration. If `cudaEventCreateWithFlags` fails
         * (returns a non-cudaSuccess status), a `cuweaver::cudaError` is thrown with contextual error information.
         *
         * @throws cuweaver::cudaError Thrown if `cudaEventCreateWithFlags` fails to create the CUDA event.
         */
        cudaEvent();

        /**
         * @brief Constructs a CUDA event with specified type-safe flags.
         *
         * @param[in] flags Type-safe configuration flags from the `cudaEventFlags` enum.
         *
         * @details Initializes the event handle to `nullptr`, converts the enum flags to the native `cudaEventFlags_t` type,
         * then creates the event via `cudaEventCreateWithFlags` with the provided flags.
         *
         * @throws cuweaver::cudaError Thrown if `cudaEventCreateWithFlags` fails to create the CUDA event.
         */
        explicit cudaEvent(cudaEventFlags flags);

        /**
         * @brief Constructs a CUDA event with raw native flags.
         *
         * @param[in] flags Raw configuration flags compatible with CUDA's native `cudaEventFlags` type.
         *
         * @details Initializes the event handle to `nullptr`, uses the provided raw flags value,
         * then creates the event via `cudaEventCreateWithFlags` with the given flags.
         *
         * @throws cuweaver::cudaError Thrown if `cudaEventCreateWithFlags` fails to create the CUDA event.
         */
        explicit cudaEvent(cudaEventFlags_t flags);

        /**
         * @brief Constructs a CUDA event by adopting an existing native handle.
         *
         * @param[in] event Existing native CUDA event handle to take ownership of.
         *
         * @details Takes ownership of the provided `cudaEvent_t` handle (the caller must not destroy it afterward).
         * Initializes flags to `cudaEventFlags::Default` since the existing handle's creation flags are not tracked.
         */
        explicit cudaEvent(cudaEvent_t event);

        cudaEvent(const cudaEvent&) = delete;

        cudaEvent& operator=(const cudaEvent&) = delete;

        /**
         * @brief Moves ownership of resources from another CUDA event instance.
         *
         * @param[in,out] other Source `cudaEvent` instance to transfer ownership from.
         *
         * @details Transfers the underlying event handle and flags from `other` to this instance.
         * The source instance is left in a valid state with a `nullptr` event handle to avoid double destruction.
         */
        cudaEvent(cudaEvent&& other) noexcept;

        /**
         * @brief Assigns ownership of resources from another CUDA event instance via move semantics.
         *
         * @param[in,out] other Source `cudaEvent` instance to transfer ownership from.
         * @return Reference to this `cudaEvent` instance after the move.
         *
         * @details Transfers the underlying event handle and flags from `other` to this instance.
         * The source instance is left with a `nullptr` event handle. Returns a reference to this instance for method chaining.
         */
        cudaEvent& operator=(cudaEvent&& other) noexcept;

        /**
         * @brief Destroys the managed CUDA event and releases resources.
         *
         * @details Checks if the underlying event handle is non-null and calls `cudaEventDestroy` to free the CUDA event resource.
         */
        ~cudaEvent();

        /**
         * @brief Retrieves the underlying native CUDA event handle.
         *
         * @return The native `cudaEvent_t` handle managed by this instance.
         *
         * @details Returns the raw handle for use with native CUDA APIs. The handle remains owned by this `cudaEvent` instance.
         */
        [[nodiscard]] cudaEvent_t nativeHandle() const noexcept;

        /**
         * @brief Retrieves the flags used to create the CUDA event.
         *
         * @return The configuration flags (as `cudaEventFlags_t`) associated with this event.
         *
         * @details Returns the flags value provided during construction (either from the enum or raw input).
         */
        [[nodiscard]] cudaEventFlags_t getFlags() const noexcept;

        /**
         * @brief Resets the managed CUDA event to a new native handle.
         *
         * @details Destroys the currently managed CUDA event (if valid, i.e., `isValid()` returns true)
         * and takes ownership of the provided native handle. The caller must ensure the input handle
         * is a valid, unmanaged `cudaEvent_t` instance.
         *
         * @param[in] event New native CUDA event handle to assume ownership of.
         *
         * @par Returns
         *     Nothing.
         */
        void reset(cudaEvent_t event = nullptr) noexcept;

        /**
         * @brief Checks if the managed CUDA event is valid.
         *
         * @details A CUDA event is considered valid if its underlying native handle is non-null (indicating
         * the event was successfully created or assigned via construction/move operations).
         *
         * @return True if the managed event handle is non-null; false otherwise.
         */
        [[nodiscard]] bool isValid() const noexcept;

    private:
        cudaEvent_t event; //!< Native CUDA event handle managed by this wrapper.
        cudaEventFlags_t flags; //!< Flags used to configure the event during creation.
    };
}

#endif //__CUDACC__

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_EVENT_CUH
