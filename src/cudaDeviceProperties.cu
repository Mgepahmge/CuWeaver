#include <cuweaver_utils/cudaDeviceProperties.cuh>

namespace cuweaver::detail {
cudaDeviceProperties::cudaDeviceProperties(const cudaDeviceProp& prop)
    : properties(prop) {}

std::string cudaDeviceProperties::getName() const noexcept {
    return properties.name;
}

std::string cudaDeviceProperties::getUuid() const noexcept {
    return properties.uuid.bytes;
}

size_t cudaDeviceProperties::getTotalGlobalMem() const noexcept {
    return properties.totalGlobalMem;
}

size_t cudaDeviceProperties::getSharedMemPerBlock() const noexcept {
    return properties.sharedMemPerBlock;
}

int cudaDeviceProperties::getRegsPerBlock() const noexcept {
    return properties.regsPerBlock;
}

int cudaDeviceProperties::getWarpSize() const noexcept {
    return properties.warpSize;
}

size_t cudaDeviceProperties::getMemPitch() const noexcept {
    return properties.memPitch;
}

int cudaDeviceProperties::getMaxThreadsPerBlock() const noexcept {
    return properties.maxThreadsPerBlock;
}

int cudaDeviceProperties::getMaxThreadsDim(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxThreadsDim[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxGridSize(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxGridSize[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getClockRate() const noexcept {
    return properties.clockRate;
}

size_t cudaDeviceProperties::getTotalConstMem() const noexcept {
    return properties.totalConstMem;
}

int cudaDeviceProperties::getMajor() const noexcept {
    return properties.major;
}

int cudaDeviceProperties::getMinor() const noexcept {
    return properties.minor;
}

size_t cudaDeviceProperties::getTextureAlignment() const noexcept {
    return properties.textureAlignment;
}

size_t cudaDeviceProperties::getTexturePitchAlignment() const noexcept {
    return properties.texturePitchAlignment;
}

int cudaDeviceProperties::getDeviceOverlap() const noexcept {
    return properties.deviceOverlap;
}

int cudaDeviceProperties::getMultiProcessorCount() const noexcept {
    return properties.multiProcessorCount;
}

int cudaDeviceProperties::getKernelExecTimeoutEnabled() const noexcept {
    return properties.kernelExecTimeoutEnabled;
}

int cudaDeviceProperties::getIntegrated() const noexcept {
    return properties.integrated;
}

int cudaDeviceProperties::getCanMapHostMemory() const noexcept {
    return properties.canMapHostMemory;
}

int cudaDeviceProperties::getComputeMode() const noexcept {
    return properties.computeMode;
}

int cudaDeviceProperties::getMaxTexture1D() const noexcept {
    return properties.maxTexture1D;
}

int cudaDeviceProperties::getMaxTexture1DMipmap() const noexcept {
    return properties.maxTexture1DMipmap;
}

int cudaDeviceProperties::getMaxTexture1DLinear() const noexcept {
    return properties.maxTexture1DLinear;
}

int cudaDeviceProperties::getMaxTexture2D(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxTexture2D[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTexture2DMipmap(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxTexture2DMipmap[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTexture2DLinear(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxTexture2DLinear[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTexture2DGather(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxTexture2DGather[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTexture3D(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxTexture3D[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTexture3DAlt(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxTexture3DAlt[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTextureCubemap() const noexcept {
    return properties.maxTextureCubemap;
}

int cudaDeviceProperties::getMaxTexture1DLayered(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxTexture1DLayered[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTexture2DLayered(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxTexture2DLayered[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxTextureCubemapLayered(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxTextureCubemapLayered[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxSurface1D() const noexcept {
    return properties.maxSurface1D;
}

int cudaDeviceProperties::getMaxSurface2D(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxSurface2D[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxSurface3D(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxSurface3D[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxSurface1DLayered(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxSurface1DLayered[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxSurface2DLayered(size_t index) const noexcept {
    if (isValidIndex(index, 3)) {
        return properties.maxSurface2DLayered[index];
    }
    return -1; // Invalid index
}

int cudaDeviceProperties::getMaxSurfaceCubemap() const noexcept {
    return properties.maxSurfaceCubemap;
}

int cudaDeviceProperties::getMaxSurfaceCubemapLayered(size_t index) const noexcept {
    if (isValidIndex(index, 2)) {
        return properties.maxSurfaceCubemapLayered[index];
    }
    return -1; // Invalid index
}

size_t cudaDeviceProperties::getSurfaceAlignment() const noexcept {
    return properties.surfaceAlignment;
}

int cudaDeviceProperties::getConcurrentKernels() const noexcept {
    return properties.concurrentKernels;
}

int cudaDeviceProperties::getECCEnabled() const noexcept {
    return properties.ECCEnabled;
}

int cudaDeviceProperties::getPciBusID() const noexcept {
    return properties.pciBusID;
}

int cudaDeviceProperties::getPciDeviceID() const noexcept {
    return properties.pciDeviceID;
}

int cudaDeviceProperties::getPciDomainID() const noexcept {
    return properties.pciDomainID;
}

int cudaDeviceProperties::getTccDriver() const noexcept {
    return properties.tccDriver;
}

int cudaDeviceProperties::getAsyncEngineCount() const noexcept {
    return properties.asyncEngineCount;
}

int cudaDeviceProperties::getUnifiedAddressing() const noexcept {
    return properties.unifiedAddressing;
}

int cudaDeviceProperties::getMemoryClockRate() const noexcept {
    return properties.memoryClockRate;
}

int cudaDeviceProperties::getMemoryBusWidth() const noexcept {
    return properties.memoryBusWidth;
}

int cudaDeviceProperties::getL2CacheSize() const noexcept {
    return properties.l2CacheSize;
}

int cudaDeviceProperties::getPersistingL2CacheMaxSize() const noexcept {
    return properties.persistingL2CacheMaxSize;
}

int cudaDeviceProperties::getMaxThreadsPerMultiProcessor() const noexcept {
    return properties.maxThreadsPerMultiProcessor;
}

int cudaDeviceProperties::getStreamPrioritiesSupported() const noexcept {
    return properties.streamPrioritiesSupported;
}

int cudaDeviceProperties::getGlobalL1CacheSupported() const noexcept {
    return properties.globalL1CacheSupported;
}

int cudaDeviceProperties::getLocalL1CacheSupported() const noexcept {
    return properties.localL1CacheSupported;
}

size_t cudaDeviceProperties::getSharedMemPerMultiprocessor() const noexcept {
    return properties.sharedMemPerMultiprocessor;
}

int cudaDeviceProperties::getRegsPerMultiprocessor() const noexcept {
    return properties.regsPerMultiprocessor;
}

int cudaDeviceProperties::getManagedMemory() const noexcept {
    return properties.managedMemory;
}

int cudaDeviceProperties::getIsMultiGpuBoard() const noexcept {
    return properties.isMultiGpuBoard;
}

int cudaDeviceProperties::getMultiGpuBoardGroupID() const noexcept {
    return properties.multiGpuBoardGroupID;
}

int cudaDeviceProperties::getSingleToDoublePrecisionPerfRatio() const noexcept {
    return properties.singleToDoublePrecisionPerfRatio;
}

int cudaDeviceProperties::getPageableMemoryAccess() const noexcept {
    return properties.pageableMemoryAccess;
}

int cudaDeviceProperties::getConcurrentManagedAccess() const noexcept {
    return properties.concurrentManagedAccess;
}

int cudaDeviceProperties::getComputePreemptionSupported() const noexcept {
    return properties.computePreemptionSupported;
}

int cudaDeviceProperties::getCanUseHostPointerForRegisteredMem() const noexcept {
    return properties.canUseHostPointerForRegisteredMem;
}

int cudaDeviceProperties::getCooperativeLaunch() const noexcept {
    return properties.cooperativeLaunch;
}

int cudaDeviceProperties::getCooperativeMultiDeviceLaunch() const noexcept {
    return properties.cooperativeMultiDeviceLaunch;
}

int cudaDeviceProperties::getPageableMemoryAccessUsesHostPageTables() const noexcept {
    return properties.pageableMemoryAccessUsesHostPageTables;
}

int cudaDeviceProperties::getDirectManagedMemAccessFromHost() const noexcept {
    return properties.directManagedMemAccessFromHost;
}

int cudaDeviceProperties::getAccessPolicyMaxWindowSize() const noexcept {
    return properties.accessPolicyMaxWindowSize;
}

cudaDeviceProp cudaDeviceProperties::nativeHandle() const noexcept{
    return properties;
}

bool cudaDeviceProperties::isValidIndex(size_t index, size_t maxIndex) {
    return index < maxIndex;
}
}
