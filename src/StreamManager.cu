#include <iostream>
#include <cuweaver/StreamManager.cuh>

#include <cuweaver/EventStreamOps.cuh>

namespace cuweaver {
    bool StreamManager::initialize(size_t numEvents, size_t numExecStreams, size_t memStreams) {
        std::lock_guard lock(managerMutex);
        if (!isCudaAvailable()) {
            throw std::runtime_error("CUDA is not available.");
        }
        clear();
        try {
            numDevices = getDeviceCount();
            const auto currentDevice = getDevice();
            streamPools.reserve(numDevices);
            eventPools.reserve(numDevices);
            readMemEventMaps.reserve(numDevices);
            writeMemEventMaps.reserve(numDevices);
            for (int i = 0; i < numDevices; ++i) {
                setDevice(i);
                streamPools.emplace_back(numExecStreams, memStreams);
                eventPools.emplace_back(numEvents);
                readMemEventMaps.emplace_back();
                writeMemEventMaps.emplace_back();
            }
            deviceEnablePeerAccessAll();
            setDevice(currentDevice);
            return true;
        }
        catch (cudaError& e) {
            std::cerr << "Failed to initialize StreamManager : " << e.what() << std::endl;
            clear();
            return false;
        }
    }

    bool StreamManager::clear() {
        try {
            streamPools.clear();
            eventPools.clear();
            readMemEventMaps.clear();
            writeMemEventMaps.clear();
            return true;
        }
        catch (cudaError& e) {
            std::cerr << "Failed to clear StreamManager : " << e.what() << std::endl;
            return false;
        }
    }

    void StreamManager::releaseEvent(dataType mem, rawEventPtr event, deviceType device, cudaMemMetaFlags flags) {
        if (!isValidDevice(device)) {
            return;
        }
        if (flags == cudaMemMetaFlags::READ) {
            readMemEventMaps[device].eraseEvent(mem, event);
        }
        else {
            writeMemEventMaps[device].eraseEvent(mem, event);
        }
    }

    void StreamManager::storeEvent(dataType mem, rawEventPtr event, deviceType device, cudaMemMetaFlags flags) {
        if (!isValidDevice(device)) {
            return;
        }
        if (flags == cudaMemMetaFlags::READ) {
            readMemEventMaps[device].recordEvent(mem, event);
        }
        else {
            writeMemEventMaps[device].recordEvent(mem, event);
        }
    }

    void StreamManager::recordMemory(dataType mem, const cudaStream& stream, const deviceType device,
                                     const cudaMemMetaFlags flags) {
        if (!isValidDevice(device)) {
            return;
        }
        try {
            TempDeviceContext tempContext(device);
            const auto& event = getEvent(device);
            eventRecord(event, stream);
            {
                std::lock_guard lock(managerMutex);
                storeEvent(mem, event.nativeHandle(), device, flags);
            }
            launchHostFunc(stream, [](void* userData) {
                const auto* deleter = static_cast<EventReleaser*>(userData);
                deleter->release();
                delete deleter;
            }, new EventReleaser(mem, event.nativeHandle(), device, flags, this));
        }
        catch (cudaError& e) {
            std::cerr << "Failed to record event in StreamManager : " << e.what() << std::endl;
        }
    }

    void StreamManager::streamWaitMemory(dataType mem, const cudaStream& stream, deviceType device,
                                         cudaMemMetaFlags flags) {
        if (!isValidDevice(device)) {
            return;
        }
        try {
            TempDeviceContext tempContext(device);
            const auto& writeEvents = writeMemEventMaps[device][mem];
            for (const auto event : writeEvents) {
                cudaStreamWaitEvent(stream.nativeHandle(), event);
            }
            if (flags == cudaMemMetaFlags::WRITE) {
                const auto& readEvents = readMemEventMaps[device][mem];
                for (const auto event : readEvents) {
                    cudaStreamWaitEvent(stream.nativeHandle(), event);
                }
            }
        }
        catch (cudaError& e) {
            std::cerr << "Failed to wait for events in StreamManager : " << e.what() << std::endl;
        }
    }

    cudaStream& StreamManager::getStream(const deviceType device, const bool isResource) {
        std::lock_guard lock(streamMutex);
        return isResource ? streamPools[device].getResourceStream() : streamPools[device].getExecutionStream();
    }

    cudaEvent& StreamManager::getEvent(const deviceType device) {
        std::lock_guard lock(eventMutex);
        return eventPools[device].acquire();
    }

    StreamManager::EventReleaser::EventReleaser(dataType mem, rawEventPtr event, deviceType device,
                                                cudaMemMetaFlags flags,
                                                StreamManager* manager) : mem(mem), event(event),
                                                                          device(device), flags(flags),
                                                                          manager(manager) {
    }

    void StreamManager::EventReleaser::release() const {
        std::lock_guard lock(manager->managerMutex);
        if (manager) {
            manager->releaseEvent(mem, event, device, flags);
            manager->eventPools[device].release(event);
        }
    }

    StreamManager::TempMemMeta::TempMemMeta(dataType mem, deviceType device,
                                            cudaMemMetaFlags flags) : mem(mem), device(device), flags(flags) {
    }

    bool StreamManager::isValidDevice(const deviceType device) const {
        if (device >= 0 && device < numDevices && device != deviceFlags::Auto && device != deviceFlags::Current) {
            return true;
        }
        return false;
    }

    StreamManager::deviceType StreamManager::getReasonableDevice(deviceType device,
                                                                 const std::vector<TempMemMeta>& tempMemMetas) {
        if (device == deviceFlags::Auto) {
            auto maxCount = 0;
            std::unordered_map<deviceType, int> deviceCounts;
            for (const auto& tempMemMeta : tempMemMetas) {
                deviceCounts[tempMemMeta.device]++;
                if (deviceCounts[tempMemMeta.device] > maxCount) {
                    maxCount = deviceCounts[tempMemMeta.device];
                    device = tempMemMeta.device;
                }
            }
        }
        if (device == deviceFlags::Current) {
            return getDeviceRaw();
        }
        return device;
    }

    void StreamManager::waitAllMem(const std::vector<TempMemMeta>& tempMemMetas, const cudaStream& stream,
                                   const deviceType device) {
        for (const auto& tempMemMeta : tempMemMetas) {
            streamWaitMemory(tempMemMeta.mem, stream, device, tempMemMeta.flags);
        }
    }

    void StreamManager::recordAllMem(const std::vector<TempMemMeta>& tempMemMetas, const cudaStream& stream,
                                     const deviceType device) {
        for (const auto& tempMemMeta : tempMemMetas) {
            recordMemory(tempMemMeta.mem, stream, device, tempMemMeta.flags);
        }
    }

    bool StreamManager::isCanAccessPeer(const deviceType device, const deviceType peerDevice) {
        if (device == peerDevice) {
            return true;
        }
        if (device == deviceFlags::Host || peerDevice == deviceFlags::Host) {
            return true;
        }
        return isDeviceCanAccessPeer(device, peerDevice);
    }

    StreamManager::deviceType StreamManager::availableMemcpyDevice(deviceType dstDevice, deviceType srcDevice) {
        if (dstDevice == deviceFlags::Host) {
            if (srcDevice == deviceFlags::Host) {
                return 0;
            }
            return srcDevice;
        }
        return dstDevice;
    }
}
