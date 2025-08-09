#include <cuweaver/Event.cuh>

namespace cuweaver {
    cudaEvent::cudaEvent() : event(nullptr), flags(static_cast<cudaEventFlags_t>(cudaEventFlags::Default)) {
        cudaEventCreate(&event, static_cast<cudaEventFlags_t>(cudaEventFlags::Default));
    }

    cudaEvent::cudaEvent(cudaEventFlags flags) : event(nullptr), flags(static_cast<cudaEventFlags_t>(flags)) {
        cudaEventCreate(&event, static_cast<cudaEventFlags_t>(flags));
    }

    cudaEvent::cudaEvent(const cudaEventFlags_t flags) : event(nullptr), flags(flags) {
        cudaEventCreate(&event, flags);
    }

    cudaEvent::cudaEvent(const cudaEvent_t event) : event(event), flags(static_cast<cudaEventFlags_t>(cudaEventFlags::Default)) {
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

    void cudaEvent::reset(const cudaEvent_t event) noexcept {
        if (this->isValid()) {
            cudaEventDestroy(this->event);
        }
        this->event = event;
    }

    bool cudaEvent::isValid() const noexcept {
        return event != nullptr;
    }
}
