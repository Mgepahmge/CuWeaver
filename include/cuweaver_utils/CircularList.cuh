/**
* @file CircularList.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Implements a circular linked list optimized for high-speed cyclic traversal.
 *
 * @details This file defines the `detail::CircularList` class, a minimalistic circular linked list
 *          designed for extreme performance in cyclic scheduling scenarios (e.g., round-robin
 *          allocation of CUDA Streams). It prioritizes fast traversal (`next()`, `get()`) and
 *          efficient node management, serving as the foundation for the CuWeaver library's
 *          CUDA Stream pool. Move semantics are supported, but copy operations are disabled.
 */
#ifndef CUWEAVER_CIRCULARLIST_CUH
#define CUWEAVER_CIRCULARLIST_CUH
#include <memory>
#include <stdexcept>

namespace cuweaver::detail {
    /**
     * @class CircularList<T> CircularList.cuh
     * @brief A circular linked list optimized for fast cyclic traversal and scheduling.
     *
     * @details This class provides a minimal API for managing a circular list, tailored for
     *          scenarios requiring repeated cyclic iteration (e.g., CUDA Stream pooling). It
     *          optimizes traversal speed by maintaining a `current` node pointer and uses
     *          in-place node construction to minimize overhead.
     *
     * @tparam T Type of data stored in the list's nodes.
     */
    template <typename T>
    class CircularList {
        /**
         * @struct Node
         * @brief Internal node structure for `CircularList`.
         *
         * @details Stores a single element of type `T` and a pointer to the next node in the
         *          circular list. Uses a variadic constructor to forward arguments to the `T`
         *          constructor for in-place initialization.
         */
        struct Node {
            T data; //!< Data stored in the node.
            Node* next; //!< Pointer to the next node in the list.
            /**
             * @brief Constructs a Node with forwarded arguments for the stored data.
             *
             * @tparam Args Types of arguments to forward to the `T` constructor.
             * @param[in] args Arguments forwarded to initialize the `data` member.
             */
            template <typename... Args>
            explicit Node(Args&&... args)
                : data(std::forward<Args>(args)...), next(nullptr) {
            }
        };

    public:
        /**
         * @brief Constructs an empty circular list.
         */
        CircularList() : head(nullptr), current(nullptr), size(0) {
        }

        /**
         * @brief Copy constructor (disabled).
         *
         * @details This class does not support copy semantics to avoid deep-copy overhead.
         */
        CircularList(const CircularList& other) = delete;
        /**
         * @brief Copy assignment operator (disabled).
         *
         * @details This class does not support copy semantics to avoid deep-copy overhead.
         */
        CircularList& operator=(const CircularList& other) = delete;
        /**
         * @brief Move constructor (noexcept).
         *
         * @details Transfers ownership of the list's nodes from `other` to this instance. The
         *          `other` list is left in an empty, valid state after the move.
         *
         * @param[in] other Rvalue reference to the list to move from.
         */
        CircularList(CircularList&& other) noexcept
            : head(other.head), current(other.current), size(other.size) {
            other.head = nullptr;
            other.current = nullptr;
            other.size = 0;
        }

        /**
         * @brief Move assignment operator (noexcept).
         *
         * @details Transfers ownership of the list's nodes from `other` to this instance. The
         *          `other` list is left in an empty, valid state after the move.
         *
         * @param[in] other Rvalue reference to the list to move from.
         * @return Reference to this list after the move.
         */
        CircularList& operator=(CircularList&& other) noexcept {
            if (this != &other) {
                head = other.head;
                current = other.current;
                size = other.size;
                other.head = nullptr;
                other.current = nullptr;
                other.size = 0;
            }
            return *this;
        }

        /**
         * @brief Clears the list, deleting all nodes and releasing memory.
         *
         * @details Iterates through the entire circular list, deletes each node, and resets
         *          the list to an empty state. Safe to call on an already empty list.
         */
        void clear() {
            if (!head) {
                return;
            }
            auto node = head;
            do {
                auto deleted = node;
                node = node->next;
                delete deleted;
            }
            while (node != head);
            head = nullptr;
            current = nullptr;
            size = 0;
        }

        /**
         * @brief Destructor.
         *
         * @details Automatically calls `clear()` to release all node memory.
         */
        ~CircularList() {
            clear();
        }

        /**
         * @brief Adds a new node to the list, constructing data in-place.
         *
         * @details Creates a new `Node` by forwarding arguments to the `T` constructor. For
         *          empty lists, initializes the circular structure (head points to itself). For
         *          non-empty lists, inserts the node after the head to maintain circularity.
         *
         * @tparam Args Types of arguments to forward to the `T` constructor.
         * @param[in] args Arguments forwarded to initialize the new node's `data` member.
         */
        template <typename... Args>
        void add(Args&&... args) {
            auto newNode = new Node(std::forward<Args>(args)...);
            if (!head) {
                head = newNode;
                head->next = head;
                current = head;
            }
            else {
                newNode->next = head->next;
                head->next = newNode;
            }
            size++;
        }

        /**
         * @brief Retrieves a reference to the current node's data.
         *
         * @details Throws an exception if the list is empty (no current node).
         *
         * @return Reference to the data in the current node.
         * @throws std::runtime_error If the list is empty.
         */
        T& get() {
            if (!current) {
                throw std::runtime_error("CircularList is empty.");
            }
            return current->data;
        }

        /**
         * @brief Moves the current node pointer to the next node (noexcept).
         *
         * @details Maintains circularity by wrapping around to the head after the last node.
         *          Does nothing if the list is empty.
         */
        void next() noexcept {
            if (!current) {
                return;
            }
            current = current->next;
        }

        /**
         * @brief Moves to the next node and retrieves its data.
         *
         * @details Combines `next()` and `get()`. Throws an exception if the list is empty.
         *
         * @return Reference to the data in the new current node.
         * @throws std::runtime_error If the list is empty.
         */
        T& getNext() {
            next();
            return get();
        }

        /**
         * @brief Checks if the list is empty (noexcept).
         *
         * @return `true` if the list has no nodes; `false` otherwise.
         */
        [[nodiscard]] bool isEmpty() const noexcept {
            return size == 0;
        }

        /**
         * @brief Returns the number of nodes in the list (noexcept).
         *
         * @return Total number of nodes stored in the list.
         */
        [[nodiscard]] size_t getSize() const noexcept {
            return size;
        }

        /**
         * @brief Iterates over all nodes, applying a function to each data element.
         *
         * @details Starts from the current node and loops through the entire circular list
         *          (one full cycle). The function `func` is invoked with a reference to each
         *          node's `data` member. Does nothing if the list is empty.
         *
         * @tparam Func Callable type that accepts a `T&` argument.
         * @param[in] func Function object to apply to each data element.
         */
        template <typename Func>
        void forEach(Func func) {
            if (!current) return;
            auto start = current;
            auto node = current;
            do {
                func(node->data);
                node = node->next;
            }
            while (node != start);
        }

        /**
         * @brief Checks if the list contains a specific value.
         *
         * @details Uses `forEach()` to search for `value` by comparing it to each node's data.
         *          Returns `true` immediately if a match is found.
         *
         * @param[in] value Value to search for in the list.
         * @return `true` if `value` exists in the list; `false` otherwise.
         */
        bool contains(const T& value) {
            bool found = false;
            forEach([&found, &value](const T& data) {
                if (data == value) {
                    found = true;
                }
            });
            return found;
        }

    private:
        Node* head; //!< Head node (entry point for the circular list).
        Node* current; //!< Current node pointer for traversal.
        size_t size; //!< Number of nodes in the list.
    };
}

#endif //CUWEAVER_CIRCULARLIST_CUH
