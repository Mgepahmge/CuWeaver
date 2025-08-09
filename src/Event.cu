#include <cuweaver/Event.cuh>
#include <cuweaver_utils/ErrorCheck.cuh>

namespace cuweaver {
    cudaEvent::cudaEvent() : event(nullptr), flags(static_cast<cudaEventFlags_t>(cudaEventFlags::Default)) {
        CUW_THROW_IF_ERROR(cudaEventCreateWithFlags(&event, static_cast<cudaEventFlags_t>(cudaEventFlags::Default)));;
    }

    cudaEvent::cudaEvent(cudaEventFlags flags) : event(nullptr), flags(static_cast<cudaEventFlags_t>(flags)) {
        CUW_THROW_IF_ERROR(cudaEventCreateWithFlags(&event, static_cast<cudaEventFlags_t>(flags)));
    }

    cudaEvent::cudaEvent(const cudaEventFlags_t flags) : event(nullptr), flags(flags) {
        CUW_THROW_IF_ERROR(cudaEventCreateWithFlags(&event, flags));
    }

    cudaEvent::cudaEvent(cudaEvent_t event) : event(event),
                                              flags(static_cast<cudaEventFlags_t>(cudaEventFlags::Default)) {
    }

    cudaEvent::cudaEvent(cudaEvent&& other) noexcept : event(other.event), flags(other.flags) {
        other.event = nullptr;
    }

    cudaEvent& cudaEvent::operator=(cudaEvent&& other) noexcept {
        if (this != &other) {
            this->reset(other.event);
            flags = other.flags;
            other.event = nullptr;
        }
        return *this;
    }

    cudaEvent::~cudaEvent() {
        if (this->isValid()) {
            cudaEventDestroy(event);
        }
    }

    cudaEvent_t cudaEvent::nativeHandle() const noexcept {
        return event;
    }

    cudaEvent::cudaEventFlags_t cudaEvent::getFlags() const noexcept {
        return flags;
    }

    void cudaEvent::reset(cudaEvent_t event) noexcept {
        if (this->isValid()) {
            cudaEventDestroy(this->event);
        }
        this->event = event;
    }

    bool cudaEvent::isValid() const noexcept {
        return event != nullptr;
    }
}
