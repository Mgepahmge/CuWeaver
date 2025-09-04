#include <cuweaver/cudaDevice.cuh>

namespace cuweaver {
    cudaDevice::cudaDevice(const int deviceId) : deviceId(deviceId) {
    }

    detail::cudaDeviceProperties cudaDevice::getProp() const {
        cudaDeviceProp prop{};
        CUW_THROW_IF_ERROR(cudaGetDeviceProperties_v2(&prop, deviceId));
        return detail::cudaDeviceProperties(prop);
    }

    void cudaDevice::setDevice() const {
        int current = 0;
        CUW_THROW_IF_ERROR(cudaGetDevice(&current));
        if (current != deviceId) {
            CUW_THROW_IF_ERROR(cudaSetDevice(deviceId));
        }
    }

    void cudaDevice::setFlags(cudaDeviceFlags flags) const {
        switchContext([flags] {
            CUW_THROW_IF_ERROR(cudaSetDeviceFlags(static_cast<unsigned int>(flags)));
        });
    }

    cudaDeviceFlags cudaDevice::getFlags() const {
        unsigned flags = 0;
        switchContext([&flags] {
            unsigned int flags_ = 0;
            CUW_THROW_IF_ERROR(cudaGetDeviceFlags(&flags_));
            flags = flags_;
        });
        return static_cast<cudaDeviceFlags>(flags);
    }

    void cudaDevice::synchronize() const {
        switchContext([] {
            CUW_THROW_IF_ERROR(cudaDeviceSynchronize());
        });
    }

    void cudaDevice::reset() const {
        switchContext([] {
            CUW_THROW_IF_ERROR(cudaDeviceReset());
        });
    }

    std::string cudaDevice::getPCIBusId(const unsigned int len) const {
        auto result = new char[len];
        CUW_THROW_IF_ERROR(cudaDeviceGetPCIBusId(result, len, deviceId));
        std::string str(result);
        delete[] result;
        return str;
    }

    int cudaDevice::getDeviceId() const noexcept {
        return deviceId;
    }

    void cudaDevice::enablePeerAccess(int peerDevice) const {
        if (peerDevice == deviceId) {
            return;
        }
        if (!isDeviceCanAccessPeer(deviceId, peerDevice)) {
            throw std::runtime_error(
                "Device " + std::to_string(deviceId) + " cannot access peer device " + std::to_string(peerDevice));
        }
        switchContext([peerDevice] {
            CUW_THROW_IF_ERROR(cudaDeviceEnablePeerAccess(peerDevice, 0));
        });
    }

    void cudaDevice::enablePeerAccess(const cudaDevice& peerDevice) const {
        enablePeerAccess(peerDevice.deviceId);
    }

    void cudaDevice::disablePeerAccess(int peerDevice) const {
        if (peerDevice == deviceId) {
            return;
        }
        if (!isDeviceCanAccessPeer(deviceId, peerDevice)) {
            throw std::runtime_error(
                "Device " + std::to_string(deviceId) + " cannot access peer device " + std::to_string(peerDevice));
        }
        switchContext([peerDevice] {
            CUW_THROW_IF_ERROR(cudaDeviceDisablePeerAccess(peerDevice));
        });
    }

    void cudaDevice::disablePeerAccess(const cudaDevice& peerDevice) const {
        disablePeerAccess(peerDevice.deviceId);
    }

    TempDeviceContext::TempDeviceContext(const int device) : originalDevice(getDeviceRaw()) {
        switchDevice(device);
    }

    TempDeviceContext::TempDeviceContext(const cudaDevice& device) : originalDevice(getDeviceRaw()) {
        switchDevice(device);
    }

    TempDeviceContext::~TempDeviceContext() {
        switchDevice(originalDevice);
    }

    int getDeviceCount() {
        int count = 0;
        CUW_THROW_IF_ERROR(cudaGetDeviceCount(&count));
        return count;
    }

    int getDeviceRaw() {
        int deviceId = 0;
        CUW_THROW_IF_ERROR(cudaGetDevice(&deviceId));
        return deviceId;
    }

    cudaDevice getDevice() {
        return cudaDevice(getDeviceRaw());
    }

    void setDevice(int deviceId) {
        CUW_THROW_IF_ERROR(cudaSetDevice(deviceId));
    }

    void setDevice(const cudaDevice& device) {
        device.setDevice();
    }

    void setValidDevices(const std::vector<int>& validDevices) {
        auto len = validDevices.size();
        if (len == 0) {
            return;
        }
        CUW_THROW_IF_ERROR(cudaSetValidDevices(const_cast<int*>(validDevices.data()), static_cast<int>(len)));
    }

    void setValidDevices(const std::initializer_list<int>& validDevices) {
        setValidDevices(std::vector(validDevices));
    }

    void setValidDevices(const std::vector<cudaDevice>& validDevices) {
        auto len = validDevices.size();
        if (len == 0) {
            return;
        }
        std::vector<int> deviceIds;
        deviceIds.reserve(validDevices.size());
        for (const auto& device : validDevices) {
            deviceIds.push_back(device.getDeviceId());
        }
        setValidDevices(deviceIds);
    }

    int chooseDeviceRaw(const cudaDeviceProp& prop) {
        int device = 0;
        CUW_THROW_IF_ERROR(cudaChooseDevice(&device, &prop));
        return device;
    }

    int chooseDeviceRaw(const detail::cudaDeviceProperties& prop) {
        return chooseDeviceRaw(prop.nativeHandle());
    }

    cudaDevice chooseDevice(const cudaDeviceProp& prop) {
        return cudaDevice(chooseDeviceRaw(prop));
    }

    cudaDevice chooseDevice(const detail::cudaDeviceProperties& prop) {
        return cudaDevice(chooseDeviceRaw(prop));
    }

    int getDeviceP2PPerformanceRank(int src, int dst) {
        int rank = 0;
        CUW_THROW_IF_ERROR(cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, src, dst));
        return rank;
    }

    int getDeviceP2PPerformanceRank(const cudaDevice& src, const cudaDevice& dst) {
        return getDeviceP2PPerformanceRank(src.getDeviceId(), dst.getDeviceId());
    }

    bool isDeviceP2PAccessSupported(int src, int dst) {
        int supported = 0;
        CUW_THROW_IF_ERROR(cudaDeviceGetP2PAttribute(&supported, cudaDevP2PAttrAccessSupported, src, dst));
        return supported;
    }

    bool isDeviceP2PAccessSupported(const cudaDevice& src, const cudaDevice& dst) {
        return isDeviceP2PAccessSupported(src.getDeviceId(), dst.getDeviceId());
    }

    bool isDeviceP2PNativeAtomicSupported(int src, int dst) {
        int supported = 0;
        CUW_THROW_IF_ERROR(cudaDeviceGetP2PAttribute(&supported, cudaDevP2PAttrNativeAtomicSupported, src, dst));
        return supported;
    }

    bool isDeviceP2PNativeAtomicSupported(const cudaDevice& src, const cudaDevice& dst) {
        return isDeviceP2PNativeAtomicSupported(src.getDeviceId(), dst.getDeviceId());
    }

    bool isDeviceP2PCudaArrayAccessSupported(int src, int dst) {
        int supported = 0;
        CUW_THROW_IF_ERROR(cudaDeviceGetP2PAttribute(&supported, cudaDevP2PAttrCudaArrayAccessSupported, src, dst));
        return supported;
    }

    bool isDeviceP2PCudaArrayAccessSupported(const cudaDevice& src, const cudaDevice& dst) {
        return isDeviceP2PCudaArrayAccessSupported(src.getDeviceId(), dst.getDeviceId());
    }

    bool isDeviceCanAccessPeer(int device, int peerDevice) {
        int canAccess = 0;
        CUW_THROW_IF_ERROR(cudaDeviceCanAccessPeer(&canAccess, device, peerDevice));
        return canAccess;
    }

    bool isDeviceCanAccessPeer(const cudaDevice& device, const cudaDevice& peerDevice) {
        return isDeviceCanAccessPeer(device.getDeviceId(), peerDevice.getDeviceId());
    }

    void deviceEnablePeerAccess(int device, int peerDevice) {
        cudaDevice temp(device);
        temp.enablePeerAccess(peerDevice);
    }

    void deviceEnablePeerAccess(const cudaDevice& device, const cudaDevice& peerDevice) {
        device.enablePeerAccess(peerDevice);
    }

    void deviceDisablePeerAccess(int device, int peerDevice) {
        cudaDevice temp(device);
        temp.disablePeerAccess(peerDevice);
    }

    void deviceDisablePeerAccess(const cudaDevice& device, const cudaDevice& peerDevice) {
        device.disablePeerAccess(peerDevice);
    }

    void deviceEnablePeerAccessEach(int device1, int device2) {
        deviceEnablePeerAccess(device1, device2);
        deviceEnablePeerAccess(device2, device1);
    }

    void deviceEnablePeerAccessEach(const cudaDevice& device1, const cudaDevice& device2) {
        deviceEnablePeerAccess(device1, device2);
        deviceEnablePeerAccess(device2, device1);
    }

    void deviceDisablePeerAccessEach(int device1, int device2) {
        deviceDisablePeerAccess(device1, device2);
        deviceDisablePeerAccess(device2, device1);
    }

    void deviceDisablePeerAccessEach(const cudaDevice& device1, const cudaDevice& device2) {
        deviceDisablePeerAccess(device1, device2);
        deviceDisablePeerAccess(device2, device1);
    }

    void deviceEnablePeerAccessAll() {
        auto deviceCount = getDeviceCount();
        for (int i = 0; i < deviceCount; i++) {
            for (int j = 0; j < deviceCount; j++) {
                if (i != j && isDeviceCanAccessPeer(i, j)) {
                    deviceEnablePeerAccess(i, j);
                }
            }
        }
    }

    void deviceDisablePeerAccessAll() {
        auto deviceCount = getDeviceCount();
        for (int i = 0; i < deviceCount; i++) {
            for (int j = 0; j < deviceCount; j++) {
                if (i != j && isDeviceCanAccessPeer(i, j)) {
                    deviceDisablePeerAccess(i, j);
                }
            }
        }
    }

    void switchDevice(const int device) {
        if (const auto current = getDeviceRaw(); current != device) {
            setDevice(device);
        }
    }

    void switchDevice(const cudaDevice& device) {
        switchDevice(device.getDeviceId());
    }
}
