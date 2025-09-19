/**
* @file StreamManager.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Declaration of the StreamManager and supporting utilities for CUDA stream and event management.
 *
 * @details This header defines the core components used by the cuweaver library to orchestrate CUDA
 * operations across one or more devices. The main class, `StreamManager`, is a singleton that
 * encapsulates pools of streams and events, memory dependency tracking, and helper functions
 * to allocate, copy and free device memory asynchronously. It integrates with other utilities
 * from the `cuweaver_utils` namespace to record and wait on memory events so that kernels and
 * memory operations can be issued safely without explicit user synchronization. Additionally,
 * this file declares internal helper structures such as `EventReleaser` and `TempMemMeta` and
 * provides a utility to unpack memory metadata wrappers. All declarations reside in the
 * `cuweaver` namespace.
 */
#ifndef CUWEAVER_STREAMMANAGER_CUH
#define CUWEAVER_STREAMMANAGER_CUH

#ifdef __CUDACC__
#include <mutex>
#include <cuweaver/StreamPool.cuh>
#include <cuweaver/EventPool.cuh>
#include <cuweaver/cudaDevice.cuh>
#include <cuweaver_utils/MemEventMap.cuh>

#include "cuweaver_utils/cudaMemMeta.cuh"

namespace cuweaver {
    namespace detail {
        /*!
         * @brief Extracts a raw pointer from a CUDA memory metadata wrapper.
         *
         * @details This helper examines the type of the provided argument at compile time. If
         * `isCudaMemMeta_v` evaluates to true for the decayed type of the argument, the function
         * returns the underlying `data` member of the metadata wrapper. Otherwise, the
         * argument is perfectly forwarded. This allows user code to pass either plain pointers
         * or metadata-wrapped pointers to kernels transparently without manual unpacking.
         *
         * @tparam T Type of the argument to unpack. May be a memory metadata wrapper or a raw pointer.
         *
         * @param[in] arg Argument that potentially contains a CUDA memory metadata wrapper.
         *
         * @return The raw data pointer extracted from the metadata, or the forwarded argument if
         * no extraction is required.
         */
        template <typename T>
        decltype(auto) unpackMemMeta(T&& arg) {
            if constexpr (isCudaMemMeta_v<std::decay_t<T>>) {
                return arg.data;
            }
            else {
                return std::forward<T>(arg);
            }
        }
    }

    /**
     * @class StreamManager [StreamManager.cuh]
     * @brief Singleton responsible for managing CUDA streams, events and memory dependencies.
     *
     * @details The `StreamManager` orchestrates CUDA operations across multiple devices by
     * maintaining pools of execution and resource streams, managing event pools for
     * synchronization, and tracking memory access with read/write event maps. It provides
     * high‑level methods to initialize resource pools, allocate and free device memory
     * asynchronously, perform memory copies between devices and host, and launch kernels with
     * automatic synchronization of memory dependencies. The class is non‑copyable and should
     * be accessed via the `getInstance()` method. Internally it uses a RAII device context
     * switcher (`TempDeviceContext`) to ensure operations execute on the correct device.
     */
    class StreamManager {
    public:
        using rawEventType = std::remove_pointer_t<cudaEvent_t>;
        //!< Alias for the underlying CUDA event type (non‑pointer).
        using rawEventPtr = cudaEvent_t; //!< Alias for a pointer to a CUDA event.
        using rawStreamType = std::remove_pointer_t<cudaStream_t>;
        //!< Alias for the underlying CUDA stream type (non‑pointer).
        using rawStreamPtr = cudaStream_t; //!< Alias for a pointer to a CUDA stream.
        using deviceType = int; //!< Device identifier used throughout the manager.
        using dataType = void*; //!< Generic pointer type representing device memory.


        /*!
         * @brief Retrieves the global StreamManager instance.
         *
         * @details The manager follows the singleton pattern. This static method lazily constructs
         * a single instance of `StreamManager` on first invocation and returns a reference to
         * it on subsequent calls.
         *
         * @par Parameters
         *     None.
         *
         * @return A reference to the unique `StreamManager` instance used throughout the application.
         */
        static StreamManager& getInstance() {
            static StreamManager instance;
            return instance;
        }

        /*!
         * @brief Constructs a new `StreamManager` instance.
         *
         * @details The constructor is defaulted and does not perform any initialization. Resource
         * pools are set up via the `initialize()` method. Instances are intended to be obtained
         * through `getInstance()` rather than constructed directly.
         */
        StreamManager() = default;

        /*!
         * @brief Deleted copy constructor.
         *
         * @details Copying a `StreamManager` is not allowed because it manages unique CUDA
         * resources and state. Use `getInstance()` to obtain a reference to the singleton.
         */
        StreamManager(const StreamManager&) = delete;

        /*!
         * @brief Deleted copy assignment operator.
         *
         * @details Assigning one `StreamManager` to another is prohibited to prevent unintended
         * sharing or duplication of underlying CUDA resources. Use the singleton instance
         * instead.
         */
        StreamManager& operator=(const StreamManager&) = delete;

        /*!
         * @brief Destroys the `StreamManager` instance.
         *
         * @details The default destructor cleans up only the object itself. Any CUDA resources
         * should be released via `clear()` prior to destruction. The destructor is trivial
         * because resource pools are stored in standard containers which handle their own
         * destruction.
         *
         * @par Returns
         *     Nothing.
         */
        ~StreamManager() = default;

        /*!
         * @brief Initializes the StreamManager and allocates resources.
         *
         * @details This method detects the number of CUDA devices available and, for each
         * device, constructs a `StreamPool` containing the specified number of execution
         * streams and memory streams, and an `EventPool` containing the specified number of
         * pre‑allocated events. It also prepares memory event maps for tracking read and
         * write dependencies and enables peer access between all devices. If initialization
         * fails, previously allocated resources are cleaned up.
         *
         * @param[in] numEvents Number of events to pre‑allocate in the pool per device.
         * @param[in] numExecStreams Number of execution streams to allocate per device.
         * @param[in] memStreams Number of resource/memory streams to allocate per device.
         *
         * @retval true Initialization completed successfully and resources were allocated.
         * @retval false A CUDA error occurred and the manager was reset to a clean state.
         */
        bool initialize(size_t numEvents, size_t numExecStreams, size_t memStreams);

        /*!
         * @brief Releases all resources managed by this StreamManager.
         *
         * @details This method clears the internal vectors of stream pools, event pools and
         * memory event maps. Any allocated CUDA streams or events are freed by their
         * respective pools. It should be called when the manager is no longer needed or
         * before re‑initialization. Errors occurring during cleanup are logged but do not
         * throw exceptions.
         *
         * @retval true All resources were released successfully.
         * @retval false An error occurred during cleanup.
         */
        bool clear();

        /*!
         * @brief Removes an event associated with a memory region and returns it to the pool.
         *
         * @details This internal helper erases the specified event from the appropriate memory
         * event map (read or write depending on `flags`) on the given device and releases
         * the event back to the `EventPool`. It is invoked by a host callback once an
         * event recorded by `recordMemory()` completes.
         *
         * @param[in] mem Pointer to the memory region whose event is to be released.
         * @param[in] event CUDA event to remove and recycle.
         * @param[in] device Identifier of the device on which the event was recorded.
         * @param[in] flags Memory access flags indicating whether the event corresponds to a
         * read (`cudaMemMetaFlags::READ`) or write (`cudaMemMetaFlags::WRITE`) operation.
         *
         * @par Returns
         *     Nothing.
         */
        void releaseEvent(dataType mem, rawEventPtr event, deviceType device, cudaMemMetaFlags flags);

        /*!
         * @brief Associates an event with a memory region in the event map.
         *
         * @details Depending on the provided flags, the event is inserted into either the
         * read or write memory event map for the given device. This enables subsequent
         * operations on the same memory region to wait for the event to complete. The
         * event itself remains allocated and will be returned to the pool via
         * `releaseEvent()` when it has finished.
         *
         * @param[in] mem Pointer to the memory region being tracked.
         * @param[in] event CUDA event to be recorded in the map.
         * @param[in] device Device identifier associated with the memory and event.
         * @param[in] flags Flags indicating whether the event represents a read or write
         * operation on the memory.
         *
         * @par Returns
         *     Nothing.
         */
        void storeEvent(dataType mem, rawEventPtr event, deviceType device, cudaMemMetaFlags flags);

        /*!
         * @brief Records a CUDA event to track access to a memory region on a given stream.
         *
         * @details The manager acquires an event from the device's `EventPool`, records it on
         * the supplied stream via `eventRecord()`, and stores the event in either the read
         * or write memory event map depending on `flags`. A host callback is then enqueued
         * on the same stream to invoke `releaseEvent()` once the event has completed,
         * returning the event to the pool and removing it from the event map. This
         * mechanism allows subsequent operations to query and wait on the last read/write
         * access to a memory region.
         *
         * @param[in] mem Pointer to the memory region that was just accessed.
         * @param[in] stream CUDA stream on which the operation took place and where the event
         * should be recorded.
         * @param[in] device Identifier of the device associated with the stream and memory.
         * @param[in] flags Memory access flags specifying whether the operation was a read or
         * write.
         *
         * @par Returns
         *     Nothing.
         */
        void recordMemory(dataType mem, const cudaStream& stream, deviceType device,
                          cudaMemMetaFlags flags);

        /*!
         * @brief Makes a stream wait for prior accesses to a memory region to complete.
         *
         * @details For the given memory pointer, this function retrieves all outstanding
         * write events from the write memory event map and inserts `cudaStreamWaitEvent`
         * calls on the provided stream to ensure it does not proceed until all writes
         * finish. If the pending operation is a write (indicated by `flags`), the
         * function also waits on all read events to guarantee exclusive access. Read
         * operations do not wait on other reads. This mechanism enforces proper
         * ordering between asynchronous operations without explicit barriers.
         *
         * @param[in] mem Pointer to the memory region that will be accessed.
         * @param[in] stream CUDA stream that should wait for prior events.
         * @param[in] device Device identifier associated with the stream and memory.
         * @param[in] flags Indicates whether the upcoming operation is a read or write
         * (`cudaMemMetaFlags::READ` or `cudaMemMetaFlags::WRITE`).
         *
         * @par Returns
         *     Nothing.
         */
        void streamWaitMemory(dataType mem, const cudaStream& stream, deviceType device, cudaMemMetaFlags flags);

        /*!
         * @brief Retrieves a CUDA stream from the pool for the specified device.
         *
         * @details Depending on the `isResource` flag, this function returns either a
         * resource stream (used primarily for memory operations) or an execution stream
         * (used for kernel launches) from the device's `StreamPool`. The returned stream
         * remains owned by the pool and should not be destroyed by the caller. Access is
         * synchronized via a mutex to ensure thread safety when multiple threads request
         * streams concurrently.
         *
         * @param[in] device Identifier of the device for which a stream is requested.
         * @param[in] isResource Optional flag indicating whether to return a resource stream
         * (`true`) or an execution stream (`false`). Defaults to execution stream.
         *
         * @return Reference to a `cudaStream` object managed by the pool for the specified device.
         */
        cudaStream& getStream(deviceType device, bool isResource = false);

        /*!
         * @brief Acquires a CUDA event from the event pool for a device.
         *
         * @details This method locks the event pool mutex, retrieves an available event
         * object from the `EventPool` associated with the specified device, and returns
         * it to the caller. The caller must record the event on a stream and eventually
         * ensure it is released back to the pool via `releaseEvent()` after completion.
         *
         * @param[in] device Identifier of the device for which an event is needed.
         *
         * @return Reference to an available `cudaEvent` object managed by the pool.
         */
        cudaEvent& getEvent(deviceType device);

        /*!
         * @brief Asynchronously allocates memory on a specified CUDA device.
         *
         * @details This template method wraps `cudaMallocAsync()` with device‑aware logic. It
         * validates the supplied device identifier, switches the current device via a
         * `TempDeviceContext`, obtains a resource stream for memory operations, and issues
         * an asynchronous allocation of `size` bytes. Once the allocation is enqueued,
         * the resulting pointer is recorded in the memory event map as a write to ensure
         * that subsequent asynchronous operations know when the memory becomes valid.
         *
         * @tparam T Element type pointed to by the allocated buffer.
         *
         * @param[out] data Address of a pointer that will receive the allocated device memory.
         * @param[in] size Number of bytes to allocate on the device.
         * @param[in] device Identifier of the device on which to perform the allocation.
         *
         * @par Returns
         *     Nothing. Errors result in no allocation and the pointer is left unchanged.
         */
        template <typename T>
        void malloc(T** data, size_t size, const deviceType device) {
            if (!isValidDevice(device)) {
                return;
            }
            TempDeviceContext tempContext(device);
            const auto& stream = getStream(device, true);
            cudaMallocAsync(data, size, stream.nativeHandle());
            recordMemory(*data, stream, device, cudaMemMetaFlags::WRITE);
        }

        /*!
         * @brief Performs an asynchronous memory copy between host and/or device pointers.
         *
         * @details This method wraps `cudaMemcpyAsync()` with additional checks and memory
         * dependency management. It first validates that the source and destination devices
         * are either valid CUDA devices or the host, and verifies that peer access is
         * available when copying between different devices. A suitable device context is
         * chosen via `availableMemcpyDevice()` (preferring the non‑host device). The
         * function then waits on any outstanding events associated with the source (read) and
         * destination (write) memory pointers to ensure data hazards are resolved. The
         * asynchronous copy is issued on a resource stream obtained for the chosen device.
         * After enqueuing the copy, both the source and destination pointers are recorded
         * in the memory event maps with appropriate read/write flags so that later
         * operations can synchronize correctly.
         *
         * @tparam T Destination element type.
         * @tparam U Source element type.
         *
         * @param[out] dst Destination pointer receiving the copied data. Must reside on `dstDevice`.
         * @param[in] dstDevice Identifier of the device associated with the destination pointer, or
         * `deviceFlags::Host` to indicate host memory.
         * @param[in] src Source pointer supplying the data. Must reside on `srcDevice`.
         * @param[in] srcDevice Identifier of the device associated with the source pointer, or
         * `deviceFlags::Host` to indicate host memory.
         * @param[in] size Number of bytes to copy.
         * @param[in] flags Kind of memory copy to perform, castable to `cudaMemcpyKind` (e.g.
         * host‑to‑device, device‑to‑host, device‑to‑device).
         *
         * @par Returns
         *     Nothing.
         *
         * @throws std::runtime_error If peer access between `dstDevice` and `srcDevice` is not enabled.
         */
        template <typename T, typename U>
        void memcpy(T* dst, const deviceType dstDevice, U* src, const deviceType srcDevice, const size_t size,
                    const memcpyFlags flags) {
            if (!(isValidDevice(dstDevice) || dstDevice == deviceFlags::Host) || !(isValidDevice(srcDevice) || srcDevice
                == deviceFlags::Host)) {
                return;
            }
            if (!isCanAccessPeer(dstDevice, srcDevice)) {
                throw std::runtime_error(
                    "Device " + std::to_string(dstDevice) + " cannot access peer device " + std::to_string(srcDevice));
            }
            const auto reasonableDevice = availableMemcpyDevice(dstDevice, srcDevice);
            TempDeviceContext tempContext(reasonableDevice);
            const auto& stream = getStream(reasonableDevice, true);
            streamWaitMemory(src, stream, reasonableDevice, cudaMemMetaFlags::READ);
            streamWaitMemory(dst, stream, reasonableDevice, cudaMemMetaFlags::WRITE);
            cudaMemcpyAsync(dst, src, size, static_cast<cudaMemcpyKind>(flags), stream.nativeHandle());
            recordMemory(dst, stream, reasonableDevice, cudaMemMetaFlags::WRITE);
            recordMemory(src, stream, reasonableDevice, cudaMemMetaFlags::READ);
        }


        /*!
         * @brief Performs an asynchronous peer‑to‑peer memory copy between two CUDA devices.
         *
         * @details This helper wraps `cudaMemcpyPeerAsync()` and adds memory dependency
         * tracking. Both source and destination devices must be valid and peer accessible.
         * The method sets the current device to `dstDevice`, obtains a resource stream,
         * waits on outstanding events for the source and destination pointers, and issues
         * the asynchronous copy. It then records read and write events for the involved
         * pointers so that subsequent operations can synchronize appropriately.
         *
         * @tparam T Destination element type.
         * @tparam U Source element type.
         *
         * @param[out] dst Destination pointer on device `dstDevice`.
         * @param[in] dstDevice Device identifier for the destination pointer.
         * @param[in] src Source pointer on device `srcDevice`.
         * @param[in] srcDevice Device identifier for the source pointer.
         * @param[in] size Number of bytes to copy.
         *
         * @par Returns
         *     Nothing. If the devices are not valid or peer access is disabled, the function returns early.
         */
        template <typename T, typename U>
        void memcpyPeer(T* dst, const deviceType dstDevice, U* src, const deviceType srcDevice, const size_t size) {
            if (!isValidDevice(dstDevice) || !isValidDevice(srcDevice)) {
                return;
            }
            TempDeviceContext tempContext(dstDevice);
            const auto& stream = getStream(dstDevice, true);
            streamWaitMemory(src, stream, dstDevice, cudaMemMetaFlags::READ);
            streamWaitMemory(dst, stream, dstDevice, cudaMemMetaFlags::WRITE);
            cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream.nativeHandle());
            recordMemory(dst, stream, dstDevice, cudaMemMetaFlags::WRITE);
            recordMemory(src, stream, dstDevice, cudaMemMetaFlags::READ);
        }

        /*!
         * @brief Asynchronously frees device memory previously allocated via this manager.
         *
         * @details Given a pointer to a device allocation, this method validates the device
         * identifier, switches to that device context, and obtains a resource stream.
         * It waits on any outstanding write events associated with the pointer to ensure
         * that all asynchronous operations using the memory have completed, and then
         * invokes `cudaFreeAsync()` to schedule the deallocation. No new event is recorded
         * because the pointer ceases to be valid after the call.
         *
         * @tparam T Element type pointed to by the allocation to free.
         *
         * @param[in,out] data Address of the pointer to device memory. The pointer value is
         * passed to `cudaFreeAsync()`; it is not modified by this function.
         * @param[in] device Identifier of the device on which the memory resides.
         *
         * @par Returns
         *     Nothing. If the device is invalid no action is taken.
         */
        template <typename T>
        void free(T** data, const deviceType device) {
            if (!isValidDevice(device)) {
                return;
            }
            TempDeviceContext tempContext(device);
            const auto& stream = getStream(device, true);
            streamWaitMemory(data, stream, device, cudaMemMetaFlags::WRITE);
            cudaFreeAsync(data, stream.nativeHandle());
        }

        /*!
         * @brief Launches a CUDA kernel on a specified device with automatic memory synchronization.
         *
         * @details This generic wrapper accepts a functor or kernel function and forwards
         * execution parameters to a standard CUDA kernel launch. Before launching, it
         * inspects each variadic argument: if an argument is a CUDA memory metadata
         * wrapper (`isCudaMemMeta_v`), the associated pointer, device and access flags are
         * collected into a temporary list. When the target device is set to
         * `deviceFlags::Auto`, the manager chooses a device based on the most frequently
         * referenced device among the metadata. If the device is `deviceFlags::Current`,
         * the current device is used. It then switches to the chosen device context,
         * obtains an execution stream, and waits on all memory dependencies via
         * `waitAllMem()`. The kernel is invoked with raw pointers extracted from the
         * metadata wrappers using `detail::unpackMemMeta()`. After the kernel is
         * enqueued, the manager records read/write events for each memory metadata via
         * `recordAllMem()` so that subsequent operations can synchronize correctly. Copying
         * metadata between devices (when kernel arguments reside on different devices) is
         * not yet implemented and is expected to be added in the future.
         *
         * @tparam Func Functor type or pointer type representing the kernel to launch.
         * @tparam Args Types of the kernel arguments. Arguments that satisfy
         * `isCudaMemMeta_v` will have their pointers unpacked and their metadata tracked.
         *
         * @param[in] kernel Functor or kernel function to invoke.
         * @param[in] grid Grid dimensions specifying the number of thread blocks.
         * @param[in] block Block dimensions specifying the number of threads per block.
         * @param[in] smem Amount of dynamic shared memory (in bytes) required by the kernel.
         * @param[in] device Device identifier on which to launch the kernel. Special values
         * such as `deviceFlags::Auto` or `deviceFlags::Current` can be used to select a
         * reasonable or current device.
         * @param[in] args Variadic list of arguments to forward to the kernel. CUDA memory
         * metadata wrappers will be unpacked to raw pointers and tracked for
         * synchronization.
         *
         * @par Returns
         *     Nothing. If the specified device is invalid no kernel is launched.
         */
        template <typename Func, typename... Args>
        void launchKernel(Func&& kernel, dim3 grid, dim3 block, size_t smem, deviceType device, Args&&... args) {
            if (!isValidDevice(device)) {
                return;
            }
            std::vector<TempMemMeta> tempMemMetas;
            constexpr size_t numMem = (0 + ... + (isCudaMemMeta_v<std::decay_t<Args>> ? 1 : 0));
            tempMemMetas.reserve(numMem);
            (collectTempMeta(tempMemMetas, args), ...);
            device = getReasonableDevice(device, tempMemMetas);
            TempDeviceContext tempContext(device);
            const auto& stream = getStream(device);
            // copy data from other devices to current device. Not implemented yet.
            waitAllMem(tempMemMetas, stream, device);
            std::forward<Func>(kernel)<<<grid, block, smem, stream.nativeHandle()>>>(
                detail::unpackMemMeta(std::forward<Args>(args))...);
            recordAllMem(tempMemMetas, stream, device);
        }

    private:
        friend struct EventReleaser;

        /**
         * @struct EventReleaser
         * @brief RAII helper used to release recorded events once they have completed.
         *
         * @details Instances of this structure are created by `recordMemory()` and passed as
         * user data to a host callback scheduled on the same stream as the event. When
         * the callback executes, it calls the `release()` method to remove the event from
         * the appropriate memory event map and return it to the event pool. This ensures
         * that events are recycled promptly without requiring the caller to explicitly
         * manage their lifetime.
         */
        struct EventReleaser {
            /*!
             * @brief Constructs an EventReleaser capturing the state needed to clean up an event.
             *
             * @param[in] mem Pointer to the memory region whose event is being managed.
             * @param[in] event Handle to the CUDA event to be released.
             * @param[in] device Device identifier on which the event was recorded.
             * @param[in] flags Memory access flags indicating whether the event corresponds
             * to a read or write operation.
             * @param[in] manager Pointer to the `StreamManager` that owns the event pool and maps.
             */
            EventReleaser(dataType mem, rawEventPtr event, deviceType device, cudaMemMetaFlags flags,
                          StreamManager* manager);

            /*!
             * @brief Releases the event and removes it from the memory event map.
             *
             * @details This method is intended to be invoked by a host callback. It locks the
             * owning `StreamManager`, erases the event from the appropriate event map,
             * returns the event to the event pool, and performs no further actions if the
             * manager pointer is null.
             *
             * @par Parameters
             *     None.
             *
             * @par Returns
             *     Nothing.
             */
            void release() const;

            dataType mem; //!< Memory address whose associated event will be cleaned up.
            rawEventPtr event; //!< Event handle that needs to be released back to the pool.
            deviceType device; //!< Device identifier associated with the event.
            cudaMemMetaFlags flags; //!< Access mode of the operation that produced the event (read or write).
            StreamManager* manager; //!< Pointer to the owning StreamManager used for cleanup.
        };

        /**
         * @struct TempMemMeta
         * @brief Lightweight descriptor capturing memory metadata for kernel launches.
         *
         * @details `TempMemMeta` holds a pointer, device identifier and access flags for a
         * memory argument passed to `launchKernel()`. These descriptors are populated
         * on the stack when scanning the variadic arguments for metadata wrappers. They
         * are then used to compute a reasonable device to launch on, to wait on
         * outstanding memory events via `waitAllMem()`, and to record new events via
         * `recordAllMem()` after the kernel is launched.
         */
        struct TempMemMeta {
            /*!
             * @brief Constructs a new temporary memory descriptor.
             *
             * @param[in] mem Pointer to the memory region being described.
             * @param[in] device Identifier of the device on which the memory resides.
             * @param[in] flags Memory access flags indicating whether the kernel will read
             * from or write to this memory region.
             */
            TempMemMeta(dataType mem, deviceType device, cudaMemMetaFlags flags);

            dataType mem; //!< Memory pointer associated with the kernel argument.
            deviceType device; //!< Device identifier for the memory pointer.
            cudaMemMetaFlags flags; //!< Intended access mode for the memory (read or write).
        };


        /*!
         * @brief Determines whether a device identifier refers to a valid CUDA device.
         *
         * @details A device is considered valid if it is a non‑negative index less than the
         * number of devices detected during initialization and is not equal to special
         * sentinel values such as `deviceFlags::Auto` or `deviceFlags::Current`. This check
         * is used throughout the manager to avoid performing operations on invalid devices.
         *
         * @param[in] device Device identifier to validate.
         *
         * @retval true The device identifier refers to a usable CUDA device.
         * @retval false The identifier is invalid or denotes a special flag.
         */
        [[nodiscard]] bool isValidDevice(deviceType device) const;

        /*!
         * @brief Extracts memory metadata from a kernel argument if applicable.
         *
         * @details When scanning the variadic arguments passed to `launchKernel()`, this
         * template helper checks whether the decayed type of the argument satisfies
         * `isCudaMemMeta_v`. If so, it constructs a `TempMemMeta` from the metadata's
         * `data`, `deviceId` and `flags` members and appends it to the provided vector.
         * Arguments that are not metadata wrappers are ignored. This function is
         * intentionally lightweight and does not perform any synchronization or allocation.
         *
         * @tparam T Type of the kernel argument being inspected.
         *
         * @param[in,out] tempMemMetas Vector used to accumulate temporary metadata descriptors.
         * @param[in] arg Kernel argument that may contain memory metadata.
         *
         * @par Returns
         *     Nothing.
         */
        template <typename T>
        void collectTempMeta(std::vector<TempMemMeta>& tempMemMetas, const T& arg) {
            using decayed = std::decay_t<T>;
            if constexpr (isCudaMemMeta_v<decayed>) {
                tempMemMetas.emplace_back(arg.data, arg.deviceId, arg.flags);
            }
        }


        /*!
         * @brief Chooses an appropriate device on which to launch a kernel or perform an operation.
         *
         * @details If the caller specifies `deviceFlags::Auto`, this function analyzes the
         * provided list of temporary memory descriptors and selects the device that
         * appears most frequently, assuming that this minimizes cross‑device transfers.
         * If the caller specifies `deviceFlags::Current`, the function returns the
         * current CUDA device as reported by `getDeviceRaw()`. Otherwise, the provided
         * device identifier is returned unchanged.
         *
         * @param[in] device Desired device identifier or special flag.
         * @param[in] tempMemMetas Collection of temporary memory descriptors used to infer
         * a reasonable device when `device` is `deviceFlags::Auto`.
         *
         * @return Selected device identifier based on the provided hints.
         */
        static deviceType getReasonableDevice(deviceType device, const std::vector<TempMemMeta>& tempMemMetas);

        /*!
         * @brief Waits on all recorded events associated with a collection of memory descriptors.
         *
         * @details Given a vector of `TempMemMeta` objects representing memory arguments to
         * a kernel launch, this function iterates through each descriptor and invokes
         * `streamWaitMemory()` for the associated pointer. This ensures that the
         * specified stream will not proceed until all outstanding read/write events on
         * those memory regions have completed. It is used internally by `launchKernel()`
         * to prepare the stream before enqueuing a kernel.
         *
         * @param[in] tempMemMetas Collection of temporary memory descriptors whose events
         * should be waited on.
         * @param[in] stream CUDA stream that will perform the waiting.
         * @param[in] device Device identifier associated with the stream and memory.
         *
         * @par Returns
         *     Nothing.
         */
        void waitAllMem(const std::vector<TempMemMeta>& tempMemMetas, const cudaStream& stream, deviceType device);

        /*!
         * @brief Records new events for a collection of memory descriptors after a kernel launch.
         *
         * @details After a kernel is enqueued on a stream, this method iterates through the
         * temporary metadata collected from the kernel arguments and calls
         * `recordMemory()` for each entry. This records a new event for each memory
         * region, marking it as being read or written by the kernel, and schedules a
         * callback to remove the event once it completes. This step is essential to
         * propagate memory dependency information to future operations.
         *
         * @param[in] tempMemMetas Collection of temporary memory descriptors corresponding
         * to kernel arguments.
         * @param[in] stream CUDA stream on which the kernel was launched.
         * @param[in] device Device identifier associated with the stream and memory.
         *
         * @par Returns
         *     Nothing.
         */
        void recordAllMem(const std::vector<TempMemMeta>& tempMemMetas, const cudaStream& stream,
                          deviceType device);

        /*!
         * @brief Checks whether one device can access the memory of another device.
         *
         * @details Peer access is implicitly allowed when both identifiers refer to the same
         * device or when either device denotes the host (`deviceFlags::Host`). For
         * distinct CUDA devices, the function calls `isDeviceCanAccessPeer()` to query
         * whether peer access has been enabled via CUDA's P2P APIs. This check is
         * performed before attempting cross‑device memory copies.
         *
         * @param[in] device Identifier of the device performing the access.
         * @param[in] peerDevice Identifier of the device whose memory may be accessed.
         *
         * @retval true Peer access between `device` and `peerDevice` is permitted.
         * @retval false Peer access is not enabled between the devices.
         */
        static bool isCanAccessPeer(deviceType device, deviceType peerDevice);

        /*!
         * @brief Determines the device on which to perform a memory copy operation.
         *
         * @details When copying between a device and the host, the copy must be issued from
         * the device side. If the destination is the host, the source device is
         * returned; otherwise the destination device is returned. This helper is used
         * internally by `memcpy()` to select the proper context for asynchronous copies.
         *
         * @param[in] dstDevice Destination device identifier or `deviceFlags::Host` for host memory.
         * @param[in] srcDevice Source device identifier or `deviceFlags::Host` for host memory.
         *
         * @return The device identifier that should own the stream used for the memory copy.
         */
        static deviceType availableMemcpyDevice(deviceType dstDevice, deviceType srcDevice);

        std::vector<StreamPool> streamPools; //!< Pools of execution and resource streams, one per device.
        std::vector<EventPool> eventPools; //!< Pools of reusable CUDA events, one per device.
        std::vector<detail::MemEventMap> readMemEventMaps;
        //!< Maps tracking the latest read events for each memory address per device.
        std::vector<detail::MemEventMap> writeMemEventMaps;
        //!< Maps tracking the latest write events for each memory address per device.
        std::mutex managerMutex; //!< Mutex protecting modifications to event maps and global state.
        std::mutex eventMutex; //!< Mutex protecting acquisition and release of events from the pools.
        std::mutex streamMutex; //!< Mutex protecting acquisition of streams from the pools.
        deviceType numDevices = 0; //!< Number of CUDA devices detected during initialization.
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_STREAMMANAGER_CUH
