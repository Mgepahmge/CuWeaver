/**
 * @file cudaDevice.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Declares the cuweaver::cudaDevice class for CUDA device management.
 *
 * @details This file provides a class to encapsulate a CUDA device by ID, enabling safe context switching,
 *          property querying, flag configuration, synchronization, and reset operations. It depends on internal utilities
 *          (detail::cudaDeviceProperties, Enum, ErrorCheck) for type safety and error handling. Compiles only
 *          when CUDA is available (__CUDACC__ defined).
 */
#ifndef CUWEAVER_CUDADEVICE_CUH
#define CUWEAVER_CUDADEVICE_CUH

#ifdef __CUDACC__
#include <vector>
#include <cuweaver_utils/cudaDeviceProperties.cuh>
#include <cuweaver_utils/Enum.cuh>
#include <cuweaver_utils/ErrorCheck.cuh>

namespace cuweaver {
    /**
     * @class cuweaver::cudaDevice
     * @brief Encapsulates a CUDA device to manage context and operations.
     *
     * @details Provides methods to initialize a device by ID, switch contexts, query hardware properties,
     *          configure scheduling flags, synchronize operations, and reset the device. Uses internal
     *          error checking (CUW_THROW_IF_ERROR) to handle CUDA API failures. Does not automatically
     *          manage context lifecycle (e.g., no implicit context switching on destruction).
     */
    class cudaDevice {
    public:
        /**
         * @brief Initializes a CUDA device object for the specified device ID.
         *
         * @param[in] deviceId 0-based ID of the CUDA device to manage.
         */
        explicit cudaDevice(int deviceId);

        /**
         * @brief Retrieves the properties of the managed CUDA device.
         *
         * @return A type-safe detail::cudaDeviceProperties object containing hardware capabilities.
         */
        [[nodiscard]] detail::cudaDeviceProperties getProp() const;

        /**
         * @brief Sets the current CUDA context to this device.
         *
         * @details Makes this device the active context for subsequent CUDA API calls.
         */
        void setDevice() const;

        /**
         * @brief Configures scheduling/behavior flags for the device.
         *
         * @param[in] flags cudaDeviceFlags to apply (e.g., synchronization policy, host memory mapping).
         */
        void setFlags(cudaDeviceFlags flags) const;

        /**
         * @brief Gets the currently active flags for the device.
         *
         * @return Current cudaDeviceFlags value for the managed device.
         */
        [[nodiscard]] cudaDeviceFlags getFlags() const;

        /**
         * @brief Blocks the host thread until the device completes all operations.
         *
         * @details Synchronizes the host with the device, ensuring all pending CUDA tasks finish before proceeding.
         */
        void synchronize() const;

        /**
         * @brief Resets the managed device to its initial state.
         *
         * @details Destroys all contexts on the device, releases allocated resources, and resets hardware state.
         */
        void reset() const;

        /**
         * @brief Retrieves the PCI bus ID string for the device.
         *
         * @param[in] len Maximum length of the buffer to store the PCI ID string.
         * @return std::string containing the device's PCI bus ID (e.g., "0000:01:00.0").
         */
        [[nodiscard]] std::string getPCIBusId(unsigned int len) const;

        /**
         * @brief Retrieves the 0-based ID of the managed CUDA device.
         *
         * @details Returns the device ID passed to the constructor of this cudaDevice object.
         *
         * @par Parameters
         *     None.
         *
         * @return 0-based integer ID of the CUDA device managed by this object.
         *
         * @note This function is noexcept and does not perform any error checking.
         */
        [[nodiscard]] int getDeviceId() const noexcept;

        /**
         * @brief Enables peer-to-peer access to the specified CUDA device.
         *
         * @details Switches the current CUDA context to this device, then enables peer access
         *          to the device with the given 0-based ID. Uses CUW_THROW_IF_ERROR to throw
         *          an exception if cudaDeviceEnablePeerAccess fails.
         *
         * @param[in] peerDevice 0-based ID of the peer CUDA device to enable access to.
         *
         * @throws std::runtime_error If enabling peer-to-peer access fails (propagated from CUDA API).
         */
        void enablePeerAccess(int peerDevice) const;

        /**
         * @brief Enables peer-to-peer access to the specified CUDA device (object overload).
         *
         * @details Overload that accepts a cudaDevice object instead of an ID. Delegates to the
         *          integer-based enablePeerAccess using the ID of the provided \p peerDevice.
         *
         * @param[in] peerDevice Reference to the cudaDevice object representing the peer device.
         *
         * @throws std::runtime_error If enabling peer-to-peer access fails (propagated from CUDA API).
         */
        void enablePeerAccess(const cudaDevice& peerDevice) const;

        /**
         * @brief Disables peer-to-peer access to the specified CUDA device.
         *
         * @details Switches the current CUDA context to this device, then disables peer access
         *          to the device with the given 0-based ID. Uses CUW_THROW_IF_ERROR to throw
         *          an exception if cudaDeviceDisablePeerAccess fails.
         *
         * @param[in] peerDevice 0-based ID of the peer CUDA device to disable access to.
         *
         * @throws std::runtime_error If disabling peer-to-peer access fails (propagated from CUDA API).
         */
        void disablePeerAccess(int peerDevice) const;

        /**
         * @brief Disables peer-to-peer access to the specified CUDA device (object overload).
         *
         * @details Overload that accepts a cudaDevice object instead of an ID. Delegates to the
         *          integer-based disablePeerAccess using the ID of the provided \p peerDevice.
         *
         * @param[in] peerDevice Reference to the cudaDevice object representing the peer device.
         *
         * @throws std::runtime_error If disabling peer-to-peer access fails (propagated from CUDA API).
         */
        void disablePeerAccess(const cudaDevice& peerDevice) const;

    private:
        /**
         * @brief Toggles the CUDA device context to execute a function, then restores the original device.
         *
         * @details Switches to the device managed by this object, runs the provided function \p func with \p args,
         *          and reverts to the previously active device. Throws an error if context switching fails.
         *
         * @tparam F Type of the function to execute on the device.
         * @tparam Args Types of arguments to pass to \p func.
         * @param[in] func Function to execute after switching to the managed device.
         * @param[in] args Arguments to forward to \p func (perfect forwarding).
         */
        template <typename F, typename... Args>
        void switchContext(F func, Args&&... args) const {
            int current = 0;
            CUW_THROW_IF_ERROR(cudaGetDevice(&current));
            if (current == deviceId) {
                func(std::forward<Args>(args)...);
                return;
            }
            CUW_THROW_IF_ERROR(cudaSetDevice(deviceId));
            func(std::forward<Args>(args)...);
            CUW_THROW_IF_ERROR(cudaSetDevice(current));
        }

        int deviceId;
    };

    /**
     * @struct TempDeviceContext
     * @brief RAII-based temporary CUDA device context switcher.
     *
     * @details Manages temporary CUDA device switching using RAII (Resource Acquisition Is Initialization).
     *          On construction, switches to the specified target device. On destruction, restores the
     *          original device that was active before this object was created.
     */
    struct TempDeviceContext {
        /**
         * @brief Constructs a context switcher targeting the specified CUDA device ID.
         *
         * @details Saves the currently active CUDA device ID (retrieved via `getDeviceRaw()`) and switches
         *          to the target device using `switchDevice()`. The original device will be restored when
         *          this object is destroyed.
         *
         * @param[in] device Target CUDA device ID to switch to.
         */
        explicit TempDeviceContext(int device);

        /**
         * @brief Constructs a context switcher targeting the device represented by a `cudaDevice` object.
         *
         * @details Saves the currently active CUDA device ID (retrieved via `getDeviceRaw()`) and switches
         *          to the device identified by the provided `cudaDevice` object (using its internal ID).
         *          The original device will be restored when this object is destroyed.
         *
         * @param[in] device `cudaDevice` object representing the target device to switch to.
         */
        explicit TempDeviceContext(const cudaDevice& device);

        /**
         * @brief Destroys the context switcher and restores the original CUDA device.
         *
         * @details Reverts the active CUDA device to the original ID saved during construction by calling
         *          `switchDevice(originalDevice)`. This operation is guaranteed to execute, even if
         *          exceptions are thrown during the object's lifetime.
         */
        ~TempDeviceContext();

        int originalDevice; //!< Original CUDA device ID to restore on destruction.
    };

    /**
     * @brief Gets the number of available CUDA-capable devices.
     *
     * @details Calls `cudaGetDeviceCount` to retrieve the total count of CUDA devices present in the system.
     *          Throws an exception if the CUDA API call fails.
     *
     * @par Parameters
     *     None.
     *
     * @return Total number of available CUDA devices (returns 0 if no devices are found).
     *
     * @throws std::runtime_error If retrieving the device count fails (propagated from CUDA API).
     */
    int getDeviceCount();

    /**
     * @brief Gets the raw 0-based ID of the currently active CUDA device.
     *
     * @details Calls `cudaGetDevice` to obtain the ID of the CUDA device currently set as active for the calling thread.
     *          Throws an exception if the CUDA API call fails.
     *
     * @par Parameters
     *     None.
     *
     * @return 0-based ID of the currently active CUDA device.
     *
     * @throws std::runtime_error If retrieving the active device ID fails (propagated from CUDA API).
     */
    int getDeviceRaw();

    /**
     * @brief Creates a `cudaDevice` object representing the currently active CUDA device.
     *
     * @details Constructs a `cudaDevice` instance using the raw device ID from `getDeviceRaw()`, encapsulating
     *          the currently active CUDA device in a type-safe wrapper.
     *
     * @par Parameters
     *     None.
     *
     * @return `cudaDevice` object for the currently active CUDA device.
     *
     * @throws std::runtime_error If retrieving the active device ID fails (propagated from `getDeviceRaw()`).
     */
    cudaDevice getDevice();

    /**
     * @brief Sets the currently active CUDA device to the specified 0-based ID.
     *
     * @details Calls `cudaSetDevice` to switch the active CUDA context to the device with the given ID.
     *          Throws an exception if the CUDA API call fails.
     *
     * @param[in] deviceId 0-based ID of the CUDA device to set as active.
     *
     * @throws std::runtime_error If switching the active device fails (propagated from CUDA API).
     */
    void setDevice(int deviceId);

    /**
     * @brief Overload to set the currently active CUDA device using a `cudaDevice` object.
     *
     * @details Delegates to the `setDevice()` method of the provided `cudaDevice` object to switch
     *          the active CUDA context to the device represented by the object.
     *
     * @param[in] device Reference to a `cudaDevice` object representing the target device.
     *
     * @throws std::runtime_error If switching the active device fails (propagated from `cudaDevice::setDevice()`).
     */
    void setDevice(const cudaDevice& device);

    /**
     * @brief Sets the list of valid CUDA devices using their 0-based IDs.
     *
     * @details Calls `cudaSetValidDevices` to configure the set of devices that subsequent CUDA context
     *          operations (e.g., `cudaSetDevice`) can use. If the input vector is empty, the function
     *          returns immediately without modifying the valid device list. Throws an exception if
     *          the CUDA API call fails.
     *
     * @param[in] validDevices Vector of 0-based CUDA device IDs to mark as valid.
     *
     * @throws std::runtime_error If setting valid devices fails (propagated from CUDA API).
     */
    void setValidDevices(const std::vector<int>& validDevices);

    /**
     * @brief Overload to set valid CUDA devices using an initializer list of 0-based IDs.
     *
     * @details Converts the initializer list to a `std::vector<int>` and delegates to the vector-based
     *          `setValidDevices` overload.
     *
     * @param[in] validDevices Initializer list of 0-based CUDA device IDs to mark as valid.
     *
     * @throws std::runtime_error If setting valid devices fails (propagated from the vector-based overload).
     */
    void setValidDevices(const std::initializer_list<int>& validDevices);

    /**
     * @brief Overload to set valid CUDA devices using a vector of `cudaDevice` objects.
     *
     * @details Extracts the 0-based ID from each `cudaDevice` in the input vector, constructs a
     *          `std::vector<int>` of device IDs, and delegates to the vector-based `setValidDevices` overload.
     *          If the input vector is empty, the function returns immediately without action.
     *
     * @param[in] validDevices Vector of `cudaDevice` objects representing devices to mark as valid.
     *
     * @throws std::runtime_error If setting valid devices fails (propagated from the vector-based overload).
     */
    void setValidDevices(const std::vector<cudaDevice>& validDevices);

    /**
     * @brief Chooses a CUDA device matching the specified properties and returns its 0-based ID.
     *
     * @details Calls `cudaChooseDevice` to select the best device matching the provided `cudaDeviceProp`
     *          filter criteria. Throws an exception if the CUDA API call fails.
     *
     * @param[in] prop CUDA device property structure defining the selection filter.
     *
     * @return 0-based ID of the CUDA device that best matches the input properties.
     *
     * @throws std::runtime_error If device selection fails (propagated from CUDA API).
     */
    int chooseDeviceRaw(const cudaDeviceProp& prop);
    /**
     * @brief Overload of `chooseDeviceRaw` that accepts a wrapped device property object.
     *
     * @details Converts the `detail::cudaDeviceProperties` object to its native `cudaDeviceProp` handle
     *          and delegates to the base `chooseDeviceRaw` overload.
     *
     * @param[in] prop Wrapped CUDA device property object defining the selection filter.
     *
     * @return 0-based ID of the CUDA device that best matches the input properties.
     *
     * @throws std::runtime_error If device selection fails (propagated from the base overload).
     */
    int chooseDeviceRaw(const detail::cudaDeviceProperties& prop);
    /**
     * @brief Chooses a CUDA device matching the specified properties and returns a `cudaDevice` wrapper.
     *
     * @details Selects a device via `chooseDeviceRaw` and encapsulates the resulting device ID in a
     *          `cudaDevice` object for type-safe access.
     *
     * @param[in] prop CUDA device property structure defining the selection filter.
     *
     * @return `cudaDevice` object representing the device that best matches the input properties.
     *
     * @throws std::runtime_error If device selection fails (propagated from `chooseDeviceRaw`).
     */
    cudaDevice chooseDevice(const cudaDeviceProp& prop);
    /**
     * @brief Overload of `chooseDevice` that accepts a wrapped device property object.
     *
     * @details Converts the `detail::cudaDeviceProperties` object to its native handle and delegates
     *          to the base `chooseDeviceRaw` overload, then wraps the result in a `cudaDevice`.
     *
     * @param[in] prop Wrapped CUDA device property object defining the selection filter.
     *
     * @return `cudaDevice` object representing the device that best matches the input properties.
     *
     * @throws std::runtime_error If device selection fails (propagated from `chooseDeviceRaw`).
     */
    cudaDevice chooseDevice(const detail::cudaDeviceProperties& prop);
    /**
     * @brief Retrieves the P2P performance rank between two CUDA devices (by 0-based ID).
     *
     * @details Calls `cudaDeviceGetP2PAttribute` to get the `cudaDevP2PAttrPerformanceRank` value,
     *          which indicates relative P2P performance (0 = highest rank, higher values = lower performance).
     *          Throws an exception if the CUDA API call fails.
     *
     * @param[in] src 0-based ID of the source CUDA device.
     * @param[in] dst 0-based ID of the destination CUDA device.
     *
     * @return Integer rank representing P2P performance (0 = best, higher = worse).
     *
     * @throws std::runtime_error If retrieving the performance rank fails (propagated from CUDA API).
     */
    int getDeviceP2PPerformanceRank(int src, int dst);
    /**
     * @brief Overload of `getDeviceP2PPerformanceRank` that accepts `cudaDevice` objects.
     *
     * @details Extracts the 0-based IDs from the input `cudaDevice` objects and delegates to the
     *          ID-based overload.
     *
     * @param[in] src `cudaDevice` object representing the source device.
     * @param[in] dst `cudaDevice` object representing the destination device.
     *
     * @return Integer rank representing P2P performance (0 = best, higher = worse).
     *
     * @throws std::runtime_error If retrieving the performance rank fails (propagated from the base overload).
     */
    int getDeviceP2PPerformanceRank(const cudaDevice& src, const cudaDevice& dst);
    /**
     * @brief Checks if peer-to-peer (P2P) memory access is supported between two CUDA devices (by ID).
     *
     * @details Calls `cudaDeviceGetP2PAttribute` to query the `cudaDevP2PAttrAccessSupported` flag.
     *          Returns `true` if P2P access is supported, `false` otherwise. Throws an exception if
     *          the CUDA API call fails.
     *
     * @param[in] src 0-based ID of the source CUDA device.
     * @param[in] dst 0-based ID of the destination CUDA device.
     *
     * @return `true` if P2P memory access is supported between the devices; `false` otherwise.
     *
     * @throws std::runtime_error If querying the P2P attribute fails (propagated from CUDA API).
     */
    bool isDeviceP2PAccessSupported(int src, int dst);
    /**
     * @brief Overload of `isDeviceP2PAccessSupported` that accepts `cudaDevice` objects.
     *
     * @details Extracts the 0-based IDs from the input `cudaDevice` objects and delegates to the
     *          ID-based overload.
     *
     * @param[in] src `cudaDevice` object representing the source device.
     * @param[in] dst `cudaDevice` object representing the destination device.
     *
     * @return `true` if P2P memory access is supported between the devices; `false` otherwise.
     *
     * @throws std::runtime_error If querying the P2P attribute fails (propagated from the base overload).
     */
    bool isDeviceP2PAccessSupported(const cudaDevice& src, const cudaDevice& dst);
    /**
     * @brief Checks if native P2P atomic operations are supported between two CUDA devices (by ID).
     *
     * @details Calls `cudaDeviceGetP2PAttribute` to query the `cudaDevP2PAttrNativeAtomicSupported` flag.
     *          Returns `true` if native atomic operations are supported over P2P, `false` otherwise.
     *          Throws an exception if the CUDA API call fails.
     *
     * @param[in] src 0-based ID of the source CUDA device.
     * @param[in] dst 0-based ID of the destination CUDA device.
     *
     * @return `true` if native P2P atomic operations are supported; `false` otherwise.
     *
     * @throws std::runtime_error If querying the P2P attribute fails (propagated from CUDA API).
     */
    bool isDeviceP2PNativeAtomicSupported(int src, int dst);
    /**
     * @brief Overload of `isDeviceP2PNativeAtomicSupported` that accepts `cudaDevice` objects.
     *
     * @details Extracts the 0-based IDs from the input `cudaDevice` objects and delegates to the
     *          ID-based overload.
     *
     * @param[in] src `cudaDevice` object representing the source device.
     * @param[in] dst `cudaDevice` object representing the destination device.
     *
     * @return `true` if native P2P atomic operations are supported; `false` otherwise.
     *
     * @throws std::runtime_error If querying the P2P attribute fails (propagated from the base overload).
     */
    bool isDeviceP2PNativeAtomicSupported(const cudaDevice& src, const cudaDevice& dst);
    /**
     * @brief Checks if P2P access to CUDA arrays is supported between two CUDA devices (by ID).
     *
     * @details Calls `cudaDeviceGetP2PAttribute` to query the `cudaDevP2PAttrCudaArrayAccessSupported` flag.
     *          Returns `true` if CUDA array access is supported over P2P, `false` otherwise. Throws an
     *          exception if the CUDA API call fails.
     *
     * @param[in] src 0-based ID of the source CUDA device.
     * @param[in] dst 0-based ID of the destination CUDA device.
     *
     * @return `true` if P2P CUDA array access is supported; `false` otherwise.
     *
     * @throws std::runtime_error If querying the P2P attribute fails (propagated from CUDA API).
     */
    bool isDeviceP2PCudaArrayAccessSupported(int src, int dst);
    /**
     * @brief Overload of `isDeviceP2PCudaArrayAccessSupported` that accepts `cudaDevice` objects.
     *
     * @details Extracts the 0-based IDs from the input `cudaDevice` objects and delegates to the
     *          ID-based overload.
     *
     * @param[in] src `cudaDevice` object representing the source device.
     * @param[in] dst `cudaDevice` object representing the destination device.
     *
     * @return `true` if P2P CUDA array access is supported; `false` otherwise.
     *
     * @throws std::runtime_error If querying the P2P attribute fails (propagated from the base overload).
     */
    bool isDeviceP2PCudaArrayAccessSupported(const cudaDevice& src, const cudaDevice& dst);

    /**
     * @brief Checks if a CUDA device can access a peer device (by 0-based IDs).
     *
     * @details Calls `cudaDeviceCanAccessPeer` to determine if the device with ID \p device
     *          can access the peer device with ID \p peerDevice. Throws an exception if the
     *          CUDA API call fails.
     *
     * @param[in] device 0-based ID of the source CUDA device.
     * @param[in] peerDevice 0-based ID of the peer CUDA device to check access for.
     *
     * @return true if the source device can access the peer device; false otherwise.
     *
     * @throws std::runtime_error If querying peer access capability fails (propagated from CUDA API).
     */
    bool isDeviceCanAccessPeer(int device, int peerDevice);
    /**
     * @brief Overload to check peer access capability using `cudaDevice` objects.
     *
     * @details Extracts 0-based IDs from the input `cudaDevice` objects and delegates to the
     *          ID-based `isDeviceCanAccessPeer` overload.
     *
     * @param[in] device Source `cudaDevice` object representing the source device.
     * @param[in] peerDevice `cudaDevice` object representing the peer device to check access for.
     *
     * @return true if the source device can access the peer device; false otherwise.
     *
     * @throws std::runtime_error If querying peer access capability fails (propagated from the base overload).
     */
    bool isDeviceCanAccessPeer(const cudaDevice& device, const cudaDevice& peerDevice);
    /**
     * @brief Enables peer-to-peer access from one CUDA device to another (by 0-based IDs).
     *
     * @details Creates a temporary `cudaDevice` object for \p device and calls its `enablePeerAccess`
     *          method to enable access to the peer device with ID \p peerDevice. Throws an exception
     *          if the operation fails.
     *
     * @param[in] device 0-based ID of the source CUDA device (to enable access from).
     * @param[in] peerDevice 0-based ID of the peer CUDA device (to enable access to).
     *
     * @throws std::runtime_error If enabling peer access fails (propagated from `cudaDevice::enablePeerAccess()`).
     */
    void deviceEnablePeerAccess(int device, int peerDevice);
    /**
     * @brief Overload to enable peer access using `cudaDevice` objects.
     *
     * @details Delegates to the `enablePeerAccess` method of the \p device object to enable
     *          access to the \p peerDevice.
     *
     * @param[in] device `cudaDevice` object representing the source device (to enable access from).
     * @param[in] peerDevice `cudaDevice` object representing the peer device (to enable access to).
     *
     * @throws std::runtime_error If enabling peer access fails (propagated from `cudaDevice::enablePeerAccess()`).
     */
    void deviceEnablePeerAccess(const cudaDevice& device, const cudaDevice& peerDevice);
    /**
     * @brief Disables peer-to-peer access from one CUDA device to another (by 0-based IDs).
     *
     * @details Creates a temporary `cudaDevice` object for \p device and calls its `disablePeerAccess`
     *          method to disable access to the peer device with ID \p peerDevice. Throws an exception
     *          if the operation fails.
     *
     * @param[in] device 0-based ID of the source CUDA device (to disable access from).
     * @param[in] peerDevice 0-based ID of the peer CUDA device (to disable access to).
     *
     * @throws std::runtime_error If disabling peer access fails (propagated from `cudaDevice::disablePeerAccess()`).
     */
    void deviceDisablePeerAccess(int device, int peerDevice);
    /**
     * @brief Overload to disable peer access using `cudaDevice` objects.
     *
     * @details Delegates to the `disablePeerAccess` method of the \p device object to disable
     *          access to the \p peerDevice.
     *
     * @param[in] device `cudaDevice` object representing the source device (to disable access from).
     * @param[in] peerDevice `cudaDevice` object representing the peer device (to disable access to).
     *
     * @throws std::runtime_error If disabling peer access fails (propagated from `cudaDevice::disablePeerAccess()`).
     */
    void deviceDisablePeerAccess(const cudaDevice& device, const cudaDevice& peerDevice);
    /**
     * @brief Enables bidirectional peer-to-peer access between two CUDA devices (by 0-based IDs).
     *
     * @details Enables peer access from \p device1 to \p device2 and from \p device2 to \p device1
     *          by calling `deviceEnablePeerAccess` twice.
     *
     * @param[in] device1 0-based ID of the first CUDA device.
     * @param[in] device2 0-based ID of the second CUDA device.
     *
     * @throws std::runtime_error If enabling peer access fails for either direction.
     */
    void deviceEnablePeerAccessEach(int device1, int device2);
    /**
     * @brief Overload to enable bidirectional peer access using `cudaDevice` objects.
     *
     * @details Enables bidirectional peer access between the two `cudaDevice` objects by calling
     *          the ID-based `deviceEnablePeerAccessEach` overload with their device IDs.
     *
     * @param[in] device1 First `cudaDevice` object.
     * @param[in] device2 Second `cudaDevice` object.
     *
     * @throws std::runtime_error If enabling peer access fails for either direction.
     */
    void deviceEnablePeerAccessEach(const cudaDevice& device1, const cudaDevice& device2);
    /**
     * @brief Disables bidirectional peer-to-peer access between two CUDA devices (by 0-based IDs).
     *
     * @details Disables peer access from \p device1 to \p device2 and from \p device2 to \p device1
     *          by calling `deviceDisablePeerAccess` twice.
     *
     * @param[in] device1 0-based ID of the first CUDA device.
     * @param[in] device2 0-based ID of the second CUDA device.
     *
     * @throws std::runtime_error If disabling peer access fails for either direction.
     */
    void deviceDisablePeerAccessEach(int device1, int device2);
    /**
     * @brief Overload to disable bidirectional peer access using `cudaDevice` objects.
     *
     * @details Disables bidirectional peer access between the two `cudaDevice` objects by calling
     *          the ID-based `deviceDisablePeerAccessEach` overload with their device IDs.
     *
     * @param[in] device1 First `cudaDevice` object.
     * @param[in] device2 Second `cudaDevice` object.
     *
     * @throws std::runtime_error If disabling peer access fails for either direction.
     */
    void deviceDisablePeerAccessEach(const cudaDevice& device1, const cudaDevice& device2);
    /**
     * @brief Enables peer-to-peer access for all pairwise accessible CUDA devices.
     *
     * @details Iterates over all CUDA devices and enables peer access between every pair of
     *          devices that can access each other (checked via `isDeviceCanAccessPeer`). Throws
     *          an exception if any CUDA API call fails.
     *
     * @par Parameters
     *     None.
     *
     * @throws std::runtime_error If enabling peer access for any device pair fails.
     */
    void deviceEnablePeerAccessAll();
    /**
     * @brief Disables peer-to-peer access for all pairwise accessible CUDA devices.
     *
     * @details Iterates over all CUDA devices and disables peer access between every pair of
     *          devices that can access each other (checked via `isDeviceCanAccessPeer`). Throws
     *          an exception if any CUDA API call fails.
     *
     * @par Parameters
     *     None.
     *
     * @throws std::runtime_error If disabling peer access for any device pair fails.
     */
    void deviceDisablePeerAccessAll();

    /**
     * @brief Switches the current CUDA device to the specified ID if not already active.
     *
     * @details Checks if the current CUDA device (retrieved via `getDeviceRaw()`) matches the target ID.
     *          If they differ, calls `setDevice()` to switch to the target device. This avoids redundant
     *          device switch operations.
     *
     * @param[in] device Target CUDA device ID.
     */
    void switchDevice(int device);

    /**
     * @brief Switches the current CUDA device using a wrapped device object.
     *
     * @details Extracts the target device ID from the provided `cudaDevice` object (via `getDeviceId()`)
     *          and forwards it to the overloaded `switchDevice()` method that accepts an integer device ID.
     *
     * @param[in] device Wrapped CUDA device object containing the target device ID.
     */
    void switchDevice(const cudaDevice& device);
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif


#endif //CUWEAVER_CUDADEVICE_CUH
