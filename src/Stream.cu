#include <cuweaver/Stream.cuh>
#include <cuweaver_utils/ErrorCheck.cuh>

namespace cuweaver {
    int cudaStream::getLeastPriority() noexcept {
        int leastPriority;
        cudaDeviceGetStreamPriorityRange(&leastPriority, nullptr);
        return leastPriority;
    }

    int cudaStream::getGreatestPriority() noexcept {
        int greatestPriority;
        cudaDeviceGetStreamPriorityRange(nullptr, &greatestPriority);
        return greatestPriority;
    }

    bool cudaStream::isPriorityValid(const cudaStreamPriority_t priority) noexcept {
        return priority <= getLeastPriority() && priority >= getGreatestPriority();
    }

    cudaStream cudaStream::defaultStream() noexcept {
        return cudaStream{nullptr};
    }

    cudaStream::cudaStream() : stream(nullptr),
                               flags(static_cast<cudaStreamFlags_t>(cudaStreamFlags::Default)),
                               priority(DefaultPriority) {
        CUW_THROW_IF_ERROR(cudaStreamCreateWithPriority(&stream, flags, priority));
    }

    cudaStream::cudaStream(cudaStreamFlags flags, const cudaStreamPriority_t priority)  : stream(nullptr),
    flags(static_cast<cudaStreamFlags_t>(flags)),
    priority(isPriorityValid(priority) ? priority : DefaultPriority) {
        CUW_THROW_IF_ERROR(cudaStreamCreateWithPriority(&stream, this->flags, this->priority));
    }

    cudaStream::cudaStream(const cudaStreamFlags_t flags, const cudaStreamPriority_t priority) : stream(nullptr),
    flags(flags),
    priority(isPriorityValid(priority) ? priority : DefaultPriority) {
        CUW_THROW_IF_ERROR(cudaStreamCreateWithPriority(&stream, this->flags, this->priority));
    }

    cudaStream::cudaStream(cudaStream_t stream) : stream(stream),
    flags(static_cast<cudaStreamFlags_t>(cudaStreamFlags::Default)),
    priority(DefaultPriority) {
    }

    cudaStream::cudaStream(cudaStream&& other) noexcept : stream(other.stream),
    flags(other.flags),
    priority(other.priority) {
        other.stream = nullptr;
    }

    cudaStream& cudaStream::operator=(cudaStream&& other) noexcept {
        if (this != &other) {
            this->reset(other.stream);
            flags = other.flags;
            priority = other.priority;
            other.stream = nullptr;
        }
        return *this;
    }

    cudaStream::~cudaStream() {
        if (this->isValid()) {
            cudaStreamDestroy(stream);
        }
    }

    cudaStream_t cudaStream::nativeHandle() const noexcept {
        return stream;
    }

    cudaStream::cudaStreamFlags_t cudaStream::getFlags() const noexcept {
        return flags;
    }

    cudaStream::cudaStreamPriority_t cudaStream::getPriority() const noexcept {
        return priority;
    }

    void cudaStream::reset(cudaStream_t stream) noexcept {
        if (this->isValid()) {
            cudaStreamDestroy(this->stream);
        }
        this->stream = stream;
    }

    bool cudaStream::isValid() const noexcept {
        return stream != nullptr;
    }
}
