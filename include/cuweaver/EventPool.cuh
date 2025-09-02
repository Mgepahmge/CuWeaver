/**
* @file EventPool.cuh
* @author Mgepahmge (https://github.com/Mgepahmge)
* @brief Declares the cuweaver::EventPool class for managing a reusable pool of CUDA events.
*
* @details This file defines the EventPool class, which optimizes CUDA event management by
*          reusing pre-allocated events instead of repeatedly calling `cudaEventCreate` and
*          `cudaEventDestroy`. It uses a doubly linked list to track free events and a hash
*          map to track busy events, with automatic expansion when the pool is exhausted.
*          Designed for performance-critical CUDA workflows where event reuse reduces overhead.
*/
#ifndef CUWEAVER_EVENTPOOL_CUH
#define CUWEAVER_EVENTPOOL_CUH

#ifdef __CUDACC__
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <cuweaver/Event.cuh>

namespace cuweaver {
    /**
     * @class EventPool EventPool.cuh
     * @brief A reusable pool of CUDA events for efficient event acquisition and release.
     *
     * @details The EventPool manages a collection of `cudaEvent` objects to minimize the cost of
     *          creating and destroying events. It maintains a doubly linked list of free events
     *          for fast acquisition and a hash map of busy events for tracking. When no free events
     *          are available, the pool automatically expands by doubling its size (or initializing
     *          to 1 if empty). All events are created with `cudaEventFlags::DisableTiming` by default
     *          to optimize synchronization performance.
     *
     * @note Events acquired from the pool must be released back to the pool to avoid resource leaks.
     */
    class EventPool {
    public:
        /**
         * @brief Constructs an EventPool with a specified initial capacity.
         *
         * @param[in] PoolSize Initial number of events to pre-allocate. If 0, the pool starts empty
         *                     and expands to 1 event on the first `acquire()` call.
         *
         * @throws std::runtime_error If creating initial CUDA events fails (propagated from `cudaEvent` construction).
         */
        explicit EventPool(size_t PoolSize);

        EventPool(const EventPool&) = delete; //!< Disable copy constructor.

        EventPool& operator=(const EventPool&) = delete; //!< Disable copy assignment.

        /**
         * @brief Constructs an EventPool by moving resources from another instance.
         *
         * @details Transfers ownership of the free event list, busy map, node storage,
         *          and pool size from `other` to this pool. The `other` instance is left
         *          in a valid but unspecified state after the move.
         *
         * @param[in] other The EventPool to move resources from.
         */
        EventPool(EventPool&& other) noexcept;

        /**
         * @brief Moves ownership of resources from another EventPool to this instance.
         *
         * @details Releases any existing resources in this pool, then transfers all
         *          resources (free list, busy map, nodes, pool size) from `other`.
         *          The `other` instance is left in a valid but unspecified state.
         *
         * @param[in] other The EventPool to move resources from.
         *
         * @return Reference to this EventPool after the move operation.
         */
        EventPool& operator=(EventPool&& other) noexcept;

        /**
         * @brief Acquires a free CUDA event from the pool.
         *
         * @details Retrieves an event from the free list. If the free list is empty, the pool expands
         *          to double its current size (or 1 if initially empty) before returning a new event.
         *          The acquired event is marked as busy and cannot be re-acquired until released.
         *
         * @return Reference to an available `cudaEvent` object.
         *
         * @throws std::runtime_error If expanding the pool or creating new CUDA events fails.
         */
        cudaEvent& acquire();

        /**
         * @brief Releases a CUDA event managed by this pool using its wrapper object.
         *
         * @details Extracts the native CUDA event handle from the provided wrapper object and forwards it
         *          to the overloaded release method that accepts a `cudaEvent_t`. This is a convenience
         *          method for working with wrapped event instances.
         *
         * @param[in] event The wrapped CUDA event object to release.
         *
         * @return True if the event was successfully released; false if the event was not managed by this pool.
         */
        bool release(const cudaEvent& event);

        /**
         * @brief Releases a native CUDA event handle back to this pool.
         *
         * @details Checks if the provided native event handle is tracked in the pool's busy set (`busyMap`).
         *          If found, the handle is removed from `busyMap`, and its associated node is appended to
         *          the end of the free list for future reuse. If the handle is not managed by this pool,
         *          the method returns false.
         *
         * @param[in] event The native CUDA event handle to release.
         *
         * @return True if the event was successfully released; false if the event was not managed by this pool.
         */
        bool release(cudaEvent_t event);

    private:
        /**
         * @struct Node
         * @brief Internal node structure for storing CUDA events and linking free/busy states.
         *
         * @details Each node contains a `cudaEvent` and pointers to the previous/next nodes in the
         *          free list. Nodes are owned by the pool's `allNodes` container to prevent memory leaks.
         */
        struct Node {
            cudaEvent event; //!< CUDA event stored in this node.
            Node* prev; //!< Pointer to the previous node in the free list.
            Node* next; //!< Pointer to the next node in the free list.
            /**
             * @brief Constructs a Node by moving a `cudaEvent`.
             * @param[in] e Rvalue reference to a `cudaEvent` to store in the node.
             */
            explicit Node(cudaEvent&& e) : event(std::move(e)), prev(nullptr), next(nullptr) {
            }

            /**
             * @brief Constructs a Node with a specified `cudaEventFlags`.
             * @param[in] flags Flags to use when creating the contained `cudaEvent`.
             */
            explicit Node(const cudaEventFlags flags) : event(flags), prev(nullptr), next(nullptr) {
            }

            /**
             * @brief Default constructor: creates a `cudaEvent` with `DisableTiming` flag.
             */
            Node() : event(cudaEventFlags::DisableTiming), prev(nullptr), next(nullptr) {
            }
        };

        /**
         * @brief Creates a new Node with a default event and appends it to the free list.
         *
         * @details Allocates a new Node (owned by `allNodes`) with a `cudaEvent` created using
         *          `cudaEventFlags::DisableTiming`, then adds it to the end of the free list.
         *
         * @throws std::runtime_error If creating the new `cudaEvent` fails.
         */
        void addNewNode();

        /**
         * @brief Expands the pool's capacity by doubling its current size (or initializing to 1 if empty).
         *
         * @details Increases the pool's maximum capacity to twice its current size (or 1 if initially empty).
         *          Creates new Nodes to fill the expanded capacity and adds them to the free list.
         *
         * @throws std::runtime_error If creating new nodes or CUDA events fails during expansion.
         */
        void expansion();

        Node* freeHead; //!< Head pointer of the doubly linked list of free events.
        Node* freeTail; //!< Tail pointer of the doubly linked list of free events.
        std::unordered_map<cudaEvent_t, Node*> busyMap; //!< Maps busy event native handles to their Node.
        std::vector<std::unique_ptr<Node>> allNodes; //!< Owns all Nodes to prevent memory leaks.
        size_t poolSize; //!< Current maximum capacity of the pool (number of pre-allocated events).
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_EVENTPOOL_CUH
