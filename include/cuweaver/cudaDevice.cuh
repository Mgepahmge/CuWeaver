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
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif


#endif //CUWEAVER_CUDADEVICE_CUH
