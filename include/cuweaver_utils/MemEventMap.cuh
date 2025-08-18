/**
* @file MemEventMap.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Defines the MemEventMap class for managing memory-address-to-CUDA-event mappings.
 *
 * @details This file declares the internal `MemEventMap` class, which tracks associations between
 *          memory addresses and CUDA events. It is used to manage event dependencies for memory
 *          operations in CUDA-accelerated applications.
 */
#ifndef CUWEAVER_MEMEVENTMAP_CUH
#define CUWEAVER_MEMEVENTMAP_CUH

#ifdef __CUDACC__

#include <cuweaver_utils/PointerSet.cuh>

namespace cuweaver::detail {
    /**
     * @class MemEventMap MemEventMap.cuh
     * @brief Manages associations between memory addresses and sets of CUDA events.
     *
     * @details Used internally to track which CUDA events are linked to specific memory addresses.
     *          Enables synchronization and ordering of memory-dependent operations by maintaining
     *          event sets per memory location.
     */
    class MemEventMap {
    public:
        using EventType = std::remove_pointer_t<cudaEvent_t>; //!< Underlying CUDA event structure (removes pointer from `cudaEvent_t`).
        using EventPtr = cudaEvent_t; //!< CUDA event handle (pointer to `EventType`).
        using SetType = PointerSet<EventType>; //!< Set of unique CUDA event structures (uses `PointerSet` for O(1) operations).
        using MemType = void*; //!< Generic memory address used as the key for event mappings.

        /**
         * @brief Records a CUDA event for a given memory address.
         *
         * @details Adds the event to the event set associated with the memory address. Fails if the event
         *          is already present for the memory.
         *
         * @param[in] mem Memory address to associate with the event.
         * @param[in] event CUDA event to record for the memory address.
         * @return True if the event was successfully added (not already present); false otherwise.
         */
        bool recordEvent(MemType mem, EventPtr event);

        /**
         * @brief Checks if a memory address has any associated events.
         *
         * @param[in] mem Memory address to check.
         * @return True if the memory address exists in the event map; false otherwise.
         */
        bool hasMem(MemType mem);

        /**
         * @brief Checks if a specific event is associated with a memory address.
         *
         * @param[in] mem Memory address to check.
         * @param[in] event CUDA event to look for.
         * @return True if the memory address exists and the event is associated with it; false otherwise.
         */
        bool hasEvent(MemType mem, EventPtr event);

        /**
         * @brief Checks if a CUDA event is associated with any memory address.
         *
         * @param[in] event CUDA event to look for.
         * @return True if the event exists in any memory's event set; false otherwise.
         */
        bool hasEvent(EventPtr event);

        /**
        * @brief Removes a CUDA event from the set of a memory address.
        *
        * @details Fails if the memory address or event is not found.
        *
        * @param[in] mem Memory address whose event set to modify.
        * @param[in] event CUDA event to remove.
        * @return True if the event was successfully removed; false otherwise.
        */
        bool eraseEvent(MemType mem, EventPtr event);

        /**
        * @brief Clears all memory-address-to-event associations.
        *
        * @par Returns
        *    Nothing.
        */
        void clear();

        /**
         * @brief Gets or creates the event set for a given memory address.
         *
         * @details Creates an empty event set if the memory address is not found.
         *
         * @param[in] mem Memory address to get the event set for.
         * @return Reference to the event set associated with the memory address.
         */
        SetType& operator[](MemType mem);
    private:
        /**
         * @brief Retrieves the event set for a memory address, creating it if necessary.
         *
         * @param[in] mem Memory address to get the event set for.
         * @return Reference to the event set associated with the memory address.
         */
        SetType& getEventSet(MemType mem);

        std::unordered_map<MemType, SetType> eventMap; //!< Maps memory addresses to sets of associated CUDA events.
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_MEMEVENTMAP_CUH