#include <cuweaver/EventPool.cuh>

namespace cuweaver {
    EventPool::EventPool(const size_t PoolSize) : freeHead(nullptr), freeTail(nullptr), poolSize(PoolSize) {
        if (poolSize) {
            allNodes.reserve(PoolSize);
            for (auto i = 0; i < PoolSize; ++i) {
                addNewNode();
            }
        }
    }

    EventPool::EventPool(EventPool&& other) noexcept : freeHead(other.freeHead),
                                                     freeTail(other.freeTail),
                                                     busyMap(std::move(other.busyMap)),
                                                     allNodes(std::move(other.allNodes)),
                                                     poolSize(other.poolSize) {
        other.freeHead = nullptr;
        other.freeTail = nullptr;
        other.poolSize = 0;
    }

    EventPool& EventPool::operator=(EventPool&& other) noexcept {
        if (this != &other) {
            freeHead = other.freeHead;
            freeTail = other.freeTail;
            busyMap = std::move(other.busyMap);
            allNodes = std::move(other.allNodes);
            poolSize = other.poolSize;

            other.freeHead = nullptr;
            other.freeTail = nullptr;
            other.poolSize = 0;
        }
        return *this;
    }

    cudaEvent& EventPool::acquire() {
        if (!freeHead) {
            expansion();
        }
        Node* node = freeHead;
        cudaEvent& event = node->event;

        if (freeHead == freeTail) {
            freeHead = nullptr;
            freeTail = nullptr;
        }
        else {
            freeHead = freeHead->next;
            if (freeHead) {
                freeHead->prev = nullptr;
            }
        }
        busyMap[event.nativeHandle()] = node;
        return event;
    }

    bool EventPool::release(const cudaEvent& event) {
        return release(event.nativeHandle());
    }

    bool EventPool::release(cudaEvent_t event) {
        const auto it = busyMap.find(event);
        if (it == busyMap.end()) {
            return false;
        }
        const auto node = it->second;
        busyMap.erase(it);
        node->prev = freeTail;
        node->next = nullptr;
        if (!freeTail) {
            freeHead = node;
        }
        else {
            freeTail->next = node;
        }
        freeTail = node;
        return true;
    }

    void EventPool::addNewNode() {
        allNodes.emplace_back(std::make_unique<Node>(cudaEventFlags::DisableTiming));
        Node* newNode = allNodes.back().get();
        if (!freeHead) {
            freeHead = newNode;
            freeTail = newNode;
        }
        else {
            freeTail->next = newNode;
            newNode->prev = freeTail;
            freeTail = newNode;
        }
    }

    void EventPool::expansion() {
        const auto oldPoolSize = poolSize;
        poolSize = oldPoolSize ? oldPoolSize * 2 : 1;
        allNodes.reserve(poolSize);
        for (auto i = oldPoolSize; i < poolSize; ++i) {
            addNewNode();
        }
    }
}
