/**
 * @file PointerSet.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief A set-like container for pointers with O(1) insert, erase, and lookup operations.
 *
 * @details Implements a pointer container that combines a `std::vector` for sequential storage and a
 *          `std::unordered_map` for index mapping. This design enables constant-time insertions,
 *          deletions, and membership checks. When an element is erased, it is replaced by the last
 *          element in the vector to avoid the O(n) cost of shifting elements. Iterators remain valid
 *          for elements not erased.
 */
#ifndef CUWEAVER_POINTERSET_CUH
#define CUWEAVER_POINTERSET_CUH
#include <cassert>
#include <vector>
#include <unordered_map>

namespace cuweaver::detail {
    /**
     * @class PointerSet<T> PointerSet.cuh
     * @brief A set-like container for pointers with fast insert, erase, and lookup operations.
     *
     * @details Provides a set interface for pointers where insertions, deletions, and membership checks
     *          operate in average O(1) time. Uses a `std::vector` to store pointers (ensuring efficient
     *          iteration) and a `std::unordered_map` to map pointers to their indices in the vector (enabling
     *          fast lookups). Erasing an element swaps it with the last element in the vector to avoid
     *          shifting costs, making deletions efficient even for large containers.
     *
     * @tparam T Type of the objects pointed to by the stored pointers.
     */
    template <typename T>
    class PointerSet {
    public:
        /**
         * @brief Inserts a pointer into the set.
         *
         * @details If the pointer is already present in the set, this method does nothing and returns false.
         *          Otherwise, the pointer is added to the end of the storage vector, and its index is recorded
         *          in the map for future lookups. Returns true if the pointer was successfully inserted.
         *
         * @param[in] ptr Pointer to insert into the set.
         * @return `true` if the pointer was inserted (not already present); `false` otherwise.
         */
        bool insert(T* ptr) {
            if (index_map_.find(ptr) != index_map_.end()) {
                return false;
            }
            std::size_t index = storage_.size();
            storage_.push_back(ptr);
            index_map_[ptr] = index;
            return true;
        }

        /**
         * @brief Erases a pointer from the set.
         *
         * @details If the pointer is not present, returns false. Otherwise, the pointer is removed by swapping
         *          it with the last element in the storage vector (if it's not already the last element),
         *          updating the index map for the swapped element, and then popping the last element from the vector.
         *          This avoids the O(n) cost of shifting elements.
         *
         * @param[in] ptr Pointer to erase from the set.
         * @return `true` if the pointer was erased (present in the set); `false` otherwise.
         */
        bool erase(T* ptr) {
            auto it = index_map_.find(ptr);
            if (it == index_map_.end()) {
                return false;
            }
            std::size_t index = it->second;
            std::size_t last_index = storage_.size() - 1;
            if (index != last_index) {
                T* last_ptr = storage_[last_index];
                storage_[index] = last_ptr;
                index_map_[last_ptr] = index;
            }
            storage_.pop_back();
            index_map_.erase(it);
            return true;
        }

        /**
         * @brief Checks if the set contains a specific pointer.
         *
         * @details Uses the index map to perform an average O(1) lookup.
         *
         * @param[in] ptr Pointer to check for membership.
         * @return `true` if the pointer is present in the set; `false` otherwise.
         */
        bool contains(T* ptr) const {
            return index_map_.find(ptr) != index_map_.end();
        }

        /**
         * @brief Gets the number of elements in the set.
         *
         * @return Number of pointers stored in the set.
         */
        [[nodiscard]] std::size_t size() const {
            return storage_.size();
        }

        /**
         * @brief Checks if the set is empty.
         *
         * @return `true` if the set contains no elements; `false` otherwise.
         */
        [[nodiscard]] bool empty() const {
            return storage_.empty();
        }

        /**
         * @brief Removes all elements from the set.
         *
         * @details Clears both the storage vector and the index map, releasing all resources.
         */
        void clear() {
            storage_.clear();
            index_map_.clear();
        }

        using iterator = typename std::vector<T*>::iterator; //!< Iterator type for mutable elements.
        using const_iterator = typename std::vector<T*>::const_iterator; //!< Iterator type for const elements.

        /**
         * @brief Gets an iterator to the first element in the set.
         *
         * @return Mutable iterator pointing to the first element.
         */
        iterator begin() { return storage_.begin(); }

        /**
         * @brief Gets an iterator to the past-the-end element in the set.
         *
         * @return Mutable iterator pointing past the last element.
         */
        iterator end() { return storage_.end(); }

        /**
         * @brief Gets a const iterator to the first element in the set.
         *
         * @return Const iterator pointing to the first element.
         */
        const_iterator begin() const { return storage_.begin(); }

        /**
         * @brief Gets a const iterator to the past-the-end element in the set.
         *
         * @return Const iterator pointing past the last element.
         */
        const_iterator end() const { return storage_.end(); }

        /**
         * @brief Gets a const iterator to the first element in the set (const-qualified).
         *
         * @return Const iterator pointing to the first element.
         */
        const_iterator cbegin() const { return storage_.cbegin(); }

        /**
         * @brief Gets a const iterator to the past-the-end element in the set (const-qualified).
         *
         * @return Const iterator pointing past the last element.
         */
        const_iterator cend() const { return storage_.cend(); }

        /**
         * @brief Accesses the pointer at the specified index.
         *
         * @details Performs a bounds check (via `assert`) to ensure the index is valid. The index corresponds
         *          to the position of the pointer in the underlying storage vector.
         *
         * @param[in] index Index of the element to access (0-based).
         * @return Pointer at the specified index.
         * @pre `index < size()` (enforced by `assert`).
         */
        T* operator[](std::size_t index) const {
            assert(index < storage_.size());
            return storage_[index];
        }

        /**
         * @brief Gets the current capacity of the storage vector.
         *
         * @details The capacity is the maximum number of elements the vector can hold without reallocating.
         *
         * @return Capacity of the underlying storage vector.
         */
        [[nodiscard]] std::size_t capacity() const {
            return storage_.capacity();
        }

        /**
        * @brief Reserves space for at least the specified number of elements.
        *
        * @details Reserves capacity in both the storage vector and the index map to avoid reallocations
        *          during subsequent insertions. This can improve performance for large containers.
        *
        * @param[in] capacity Minimum number of elements to reserve space for.
        */
        void reserve(std::size_t capacity) {
            storage_.reserve(capacity);
            index_map_.reserve(capacity);
        }

    private:
        std::vector<T*> storage_; //!< Sequential storage for pointers (enables efficient iteration).
        std::unordered_map<T*, std::size_t> index_map_; //!< Maps pointers to their indices in `storage_` for O(1) lookups.
    };
}

#endif //CUWEAVER_POINTERSET_CUH
