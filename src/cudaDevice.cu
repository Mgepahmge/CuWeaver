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
}
