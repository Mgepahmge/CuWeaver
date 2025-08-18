#include <algorithm>
#include <cuweaver_utils/MemEventMap.cuh>

namespace cuweaver::detail {
    bool MemEventMap::recordEvent(MemType mem, EventPtr event) {
        return getEventSet(mem).insert(event);
    }

    bool MemEventMap::hasMem(MemType mem) {
        return eventMap.find(mem) != eventMap.end();
    }

    bool MemEventMap::hasEvent(MemType mem, EventPtr event) {
        return hasMem(mem) && getEventSet(mem).contains(event);
    }

    bool MemEventMap::hasEvent(EventPtr event) {
        return std::any_of(eventMap.begin(), eventMap.end(), [event](const auto& pair) -> bool {
            return pair.second.contains(event);
        });
    }

    bool MemEventMap::eraseEvent(MemType mem, EventPtr event) {
        return getEventSet(mem).erase(event);
    }

    void MemEventMap::clear() {
        eventMap.clear();
    }

    MemEventMap::SetType& MemEventMap::operator[](MemType mem) {
        return getEventSet(mem);
    }

    MemEventMap::SetType& MemEventMap::getEventSet(MemType mem) {
        auto it = eventMap.find(mem);
        if (it != eventMap.end()) {
            return it->second;
        }
        SetType newSet;
        eventMap[mem] = newSet;
        return eventMap[mem];
    }
}
