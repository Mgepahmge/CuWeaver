/**
* @file AllocatorTraits.cuh
* @author Mgepahmge (https://github.com/Mgepahmge)
* @brief Defines allocator traits for the CUDA Weaver library to unify allocator interfaces and provide default behaviors.
*
* @details This header provides the `cudaAllocatorTraits` class, which serves as a traits type for CUDA-compatible
*          allocators. It extracts common type information (e.g., `valueType`, `pointer`, `sizeType`) from an allocator
*          and supplies default implementations for core operations like allocation, deallocation, object construction,
*          and destruction. The traits class ensures consistency across different allocators while allowing customization
*          via allocator-specific member functions.
*
*          The header relies on C++ Standard Library components (e.g., `std::type_traits`, `std::numeric_limits`) and
*          is only compiled when the CUDA compiler (`__CUDACC__`) is active. If CUDA is unavailable, a warning is emitted
*          and the header contents are skipped.
*/
#ifndef CUWEAVER_ALLOCATORTRAITS_CUH
#define CUWEAVER_ALLOCATORTRAITS_CUH

#ifdef __CUDACC__
#include <type_traits>
#include <memory>
#include <limits>
#include <stdexcept>

namespace cuweaver {
    namespace detail {
        /**
         * @brief Utility alias for `void`, used to trigger SFINAE by validating type expressions.
         *
         * @details Collapses any number of type parameters into `void`. Used in partial specializations
         *          to detect if a template instantiation (e.g., `Op<Args...>`) is well-formed.
         * @tparam ... Variadic type parameters (ignored, but drive substitution).
         */
        template <typename...>
        using void_t = void;

        /**
         * @brief Primary template for the detector pattern, defaulting to invalid operations.
         *
         * @details Base case: used when `Op<Args...>` is not a valid instantiation. Returns
         *          `std::false_type` and uses `Default` as the associated type.
         * @tparam Default Fallback type when the operation is invalid.
         * @tparam AlwaysVoid Unused placeholder for SFINAE triggering.
         * @tparam Op Template template parameter representing the operation to validate.
         * @tparam Args Type arguments to pass to `Op`.
         */
        template <typename Default, typename AlwaysVoid, template<typename...> typename Op, typename... Args>
        struct detector {
            using value_t = std::false_type;
            using type = Default;
        };

        /**
         * @brief Partial specialization of `detector` for valid operations.
         *
         * @details Specialized when `Op<Args...>` is a valid type (via `void_t<Op<Args...>>`). Returns
         *          `std::true_type` and uses `Op<Args...>` as the associated type.
         * @tparam Default Unused (superseded by valid operation type).
         * @tparam Op Template template parameter representing the valid operation.
         * @tparam Args Type arguments for which `Op<Args...>` is valid.
         */
        template <typename Default, template<typename...> typename Op, typename... Args>
        struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
            using value_t = std::true_type;
            using type = Op<Args...>;
        };

        /**
         * @brief Detects if an allocator has a valid `construct` member function.
         *
         * @details Overload resolution helper: returns `std::true_type` if `Alloc::construct(T*, Args...)`
         *          is callable. The `int` parameter is a placeholder to prioritize this overload.
         * @tparam Alloc Allocator type to check.
         * @tparam T Type of object to construct.
         * @tparam Args Types of arguments for the object's constructor.
         * @param[in] int Placeholder parameter for overload resolution (pass `0` to invoke).
         * @return `std::true_type` if `Alloc::construct` is valid; otherwise, SFINAE falls back to the other overload.
         */
        template <typename Alloc, typename T, typename... Args>
        auto test_construct(int) -> decltype(
            std::declval<Alloc&>().construct(std::declval<T*>(), std::declval<Args>()...),
            std::true_type{}
        );

        /**
         * @brief Fallback overload for `test_construct` when `Alloc::construct` is invalid.
         *
         * @details Returns `std::false_type` if the allocator lacks a valid `construct` member.
         * @tparam Alloc Allocator type to check.
         * @tparam T Type of object to construct (unused).
         * @tparam Args Constructor argument types (unused).
         * @return `std::false_type` always.
         */
        template <typename Alloc, typename T, typename... Args>
        std::false_type test_construct(...);

        /**
         * @brief Detects if an allocator has a valid `destroy` member function.
         *
         * @details Overload resolution helper: returns `std::true_type` if `Alloc::destroy(T*)`
         *          is callable. The `int` parameter is a placeholder to prioritize this overload.
         * @tparam Alloc Allocator type to check.
         * @tparam T Type of object to destroy.
         * @param[in] int Placeholder parameter for overload resolution (pass `0` to invoke).
         * @return `std::true_type` if `Alloc::destroy` is valid; otherwise, SFINAE falls back to the other overload.
         */
        template <typename Alloc, typename T>
        auto test_destroy(int) -> decltype(
            std::declval<Alloc&>().destroy(std::declval<T*>()),
            std::true_type{}
        );

        /**
         * @brief Fallback overload for `test_destroy` when `Alloc::destroy` is invalid.
         *
         * @details Returns `std::false_type` if the allocator lacks a valid `destroy` member.
         * @tparam Alloc Allocator type to check.
         * @tparam T Type of object to destroy (unused).
         * @return `std::false_type` always.
         */
        template <typename Alloc, typename T>
        std::false_type test_destroy(...);

        /**
         * @brief Detects if an allocator has a valid `memcpy` member function.
         *
         * @details Overload resolution helper: returns `std::true_type` if `Alloc::memcpy(T*, T*, sizeType, cudaMemcpyKind)`
         *          is callable. The `int` parameter is a placeholder to prioritize this overload.
         * @tparam Alloc Allocator type to check.
         * @tparam T Type of pointers for the memory copy.
         * @param[in] int Placeholder parameter for overload resolution (pass `0` to invoke).
         * @return `std::true_type` if `Alloc::memcpy` is valid; otherwise, SFINAE falls back to the other overload.
         */
        template <typename Alloc, typename T>
        auto test_memcpy(int) -> decltype(
            std::declval<Alloc&>().memcpy(
                std::declval<T*>(),
                std::declval<T*>(),
                std::declval<typename Alloc::sizeType>(),
                std::declval<cudaMemcpyKind>()
            ),
            std::true_type{}
        );

        /**
         * @brief Fallback overload for `test_memcpy` when `Alloc::memcpy` is invalid.
         *
         * @details Returns `std::false_type` if the allocator lacks a valid `memcpy` member.
         * @tparam Alloc Allocator type to check.
         * @tparam T Type of pointers (unused).
         * @return `std::false_type` always.
         */
        template <typename Alloc, typename T>
        std::false_type test_memcpy(...);

        /**
         * @brief Detects if an allocator has a valid `maxSize` member function.
         *
         * @details Overload resolution helper: returns `std::true_type` if `Alloc::maxSize()`
         *          is callable on a const instance. The `int` parameter is a placeholder to prioritize this overload.
         * @tparam Alloc Allocator type to check.
         * @param[in] int Placeholder parameter for overload resolution (pass `0` to invoke).
         * @return `std::true_type` if `Alloc::maxSize` is valid; otherwise, SFINAE falls back to the other overload.
         */
        template <typename Alloc>
        auto test_max_size(int) -> decltype(
            std::declval<const Alloc&>().maxSize(),
            std::true_type{}
        );

        /**
         * @brief Fallback overload for `test_max_size` when `Alloc::maxSize` is invalid.
         *
         * @details Returns `std::false_type` if the allocator lacks a valid `maxSize` member.
         * @tparam Alloc Allocator type to check.
         * @return `std::false_type` always.
         */
        template <typename Alloc>
        std::false_type test_max_size(...);
    }

    /**
     * @brief Type alias for the detector's boolean result, indicating template operation validity.
     *
     * @details Shorthand for `detail::detector<void, void, Op, Args...>::value_t`. Resolves to
     *          `std::true_type` if `Op<Args...>` is a valid instantiation, or `std::false_type` otherwise.
     * @tparam Op Template template parameter representing the operation to validate (e.g., a trait).
     * @tparam Args Type arguments to pass to `Op`.
     */
    template <template<typename...> typename Op, typename... Args>
    using is_detected = typename detail::detector<void, void, Op, Args...>::value_t;

    /**
     * @brief Constexpr boolean value indicating if a template operation is valid.
     *
     * @details Direct boolean alias for `is_detected<Op, Args...>::value`. Evaluates to `true` if
     *          `Op<Args...>` is a valid instantiation, `false` otherwise.
     * @tparam Op Template template parameter representing the operation to validate.
     * @tparam Args Type arguments to pass to `Op`.
     * @value `true` if `Op<Args...>` is valid; `false` otherwise.
     */
    template <template<typename...> typename Op, typename... Args>
    constexpr bool is_detected_v = is_detected<Op, Args...>::value;

    /**
    * @class cudaAllocatorTraits [AllocatorTraits.cuh]
    * @brief Traits class for CUDA-compatible allocators, unifying interface access and default behaviors.
    *
    * @details Provides type aliases for common allocator-related types (e.g., pointers, sizes) and static methods
    *          for core allocator operations (allocation, deallocation, object construction/destruction). Uses SFINAE
    *          via the detector pattern to check for allocator-specific member functions, falling back to standard
    *          implementations (e.g., placement new for construction) if the allocator does not provide them.
    *
    * @tparam Alloc The CUDA-compatible allocator type to extract traits from.
    */
    template <typename Alloc>
    class cudaAllocatorTraits {
    public:
        using allocatorType = Alloc; //!< The underlying allocator type.
        using valueType = typename Alloc::valueType; //!< The type of values allocated by the allocator.
        using pointer = typename Alloc::pointer; //!< Pointer type used by the allocator to reference values.
        using constPointer = typename Alloc::constPointer; //!< Const pointer type for immutable value references.
        using voidPointer = typename Alloc::voidPointer; //!< Void pointer type for untyped memory references.
        using constVoidPointer = typename Alloc::constVoidPointer;
        //!< Const void pointer type for immutable untyped references.
        using differenceType = typename Alloc::differenceType;
        //!< Type for representing pointer differences (e.g., offsets).
        using sizeType = typename Alloc::sizeType; //!< Type for representing memory sizes and element counts.

        /**
         * @brief Type trait indicating if the allocator has a valid `construct` member function for constructing `T` objects.
         *
         * @details Checks if the allocator provides a callable `construct(T*, Args...)` member function that can initialize
         *          an object of type `T` at the given pointer with the provided arguments. Uses `detail::test_construct`
         *          and overload resolution to detect validity.
         * @tparam T Type of the object to be constructed by the allocator's `construct` method.
         * @tparam Args Types of arguments to pass to the `T` object's constructor via `construct`.
         */
        template <typename T, typename... Args>
        using hasConstruct = decltype(detail::test_construct<Alloc, T, Args...>(0));

        /**
         * @brief Type trait indicating if the allocator has a valid `destroy` member function for destroying `T` objects.
         *
         * @details Checks if the allocator provides a callable `destroy(T*)` member function that can destroy an object
         *          of type `T` at the given pointer. Uses `detail::test_destroy` and overload resolution to detect validity.
         * @tparam T Type of the object to be destroyed by the allocator's `destroy` method.
         */
        template <typename T>
        using hasDestroy = decltype(detail::test_destroy<Alloc, T>(0));

        /**
         * @brief Type trait indicating if the allocator has a valid `memcpy` member function for copying `T` elements.
         *
         * @details Checks if the allocator provides a callable `memcpy(T*, T*, sizeType, cudaMemcpyKind)` member function
         *          for copying `n` elements of type `T` between memory regions, with a specified CUDA copy kind. Uses
         *          `detail::test_memcpy` and overload resolution to detect validity.
         * @tparam T Type of the elements to be copied by the allocator's `memcpy` method.
         */
        template <typename T>
        using hasMemcpy = decltype(detail::test_memcpy<Alloc, T>(0));

        /**
         * @brief Type trait indicating if the allocator has a valid const `maxSize` member function.
         *
         * @details Checks if the allocator provides a callable const `maxSize()` member function that returns the maximum
         *          number of elements it can allocate. Uses `detail::test_max_size` and overload resolution to detect validity.
         */
        using hasMaxSize = decltype(detail::test_max_size<Alloc>(0));

        /**
         * @brief Allocates memory using the provided allocator.
         *
         * @param[in] alloc Reference to the allocator to use for memory allocation.
         * @param[in] n Number of elements of type `valueType` to allocate space for.
         * @return Pointer to the allocated memory.
         */
        static pointer allocate(Alloc& alloc, sizeType n) {
            return alloc.allocate(n);
        }

        /**
         * @brief Deallocates memory using the provided allocator.
         *
         * @param[in] alloc Reference to the allocator to use for memory deallocation.
         * @param[in] p Pointer to the memory to deallocate.
         * @param[in] n Number of elements of type `valueType` that were originally allocated.
         * @par Returns
         *      Nothing.
         */
        static void deallocate(Alloc& alloc, pointer p, sizeType n) {
            alloc.deallocate(p, n);
        }

        /**
         * @brief Constructs an object in place using the allocator's `construct` method (if available) or placement new.
         *
         * @details Uses the allocator's `construct` member function if it exists (detected via `hasConstruct`). Otherwise,
         *          falls back to placement new to construct the object with the provided arguments.
         *
         * @tparam T Type of the object to construct.
         * @tparam Args Types of the arguments to pass to the object's constructor.
         * @param[in] alloc Reference to the allocator (used only if it provides `construct`).
         * @param[in] p Pointer to the memory location where the object will be constructed.
         * @param[in] args Forwarded arguments to pass to the object's constructor.
         * @par Returns
         *      Nothing.
         */
        template <typename T, typename... Args>
        static void construct(Alloc& alloc, T* p, Args&&... args) {
            if constexpr (hasConstruct<T, Args...>::value) {
                alloc.construct(p, std::forward<Args>(args)...);
            }
            else {
                new(static_cast<void*>(p)) T(std::forward<Args>(args)...);
            }
        }

        /**
        * @brief Destroys an object using the allocator's `destroy` method (if available) or direct destructor call.
        *
        * @details Uses the allocator's `destroy` member function if it exists (detected via `hasDestroy`). Otherwise,
        *          directly calls the object's destructor.
        *
        * @tparam T Type of the object to destroy.
        * @tparam Args Unused template parameter (retained for consistency).
        * @param[in] alloc Reference to the allocator (used only if it provides `destroy`).
        * @param[in] p Pointer to the object to destroy.
        * @par Returns
        *      Nothing.
        */
        template <typename T>
        static void destroy(Alloc& alloc, T* p) {
            if constexpr (hasDestroy<T>::value) {
                alloc.destroy(p);
            }
            else {
                p->~T();
            }
        }

        /**
         * @brief Copies memory using the allocator's `memcpy` method (if available) or `std::memcpy`.
         *
         * @details Uses the allocator's `memcpy` member function if it exists (detected via `hasMemcpy<T>`). If the allocator
         *          does not provide `memcpy`, falls back to `std::memcpy`, where the total byte count is calculated as
         *          `n * sizeof(T)`. The `cudaMemcpyKind` parameter is ignored in the fallback path.
         *
         * @tparam T Type of the elements to copy between memory regions.
         *
         * @param[in] alloc Pointer to the allocator; used only if the allocator implements a valid `memcpy` method.
         * @param[in,out] dst Pointer to the destination memory region where data will be written.
         * @param[in] src Pointer to the source memory region from which data will be read.
         * @param[in] n Number of elements of type `T` to copy (not bytes).
         * @param[in] kind CUDA memory copy type (e.g., host-to-device, device-to-host); ignored if using `std::memcpy`.
         *
         * @par Returns
         *      Nothing.
         */
        template <typename T>
        static void memcpy(Alloc& alloc, T* dst, T* src, sizeType n, cudaMemcpyKind kind) {
            if constexpr (hasMemcpy<T>::value) {
                alloc.memcpy(dst, src, n, kind);
            }
            else {
                std::memcpy(dst, src, n * sizeof(T));
            }
        }

        /**
         * @brief Gets the maximum number of elements the allocator can allocate, using the allocator's `maxSize` method (if available) or a default calculation.
         *
         * @details Uses the allocator's `maxSize` member function if it exists (detected via `hasMaxSize`). Otherwise,
         *          calculates the maximum size as `std::numeric_limits<sizeType>::max() / sizeof(valueType)`.
         *
         * @param[in] alloc Const reference to the allocator (used only if it provides `maxSize`).
         * @return Maximum number of elements of type `valueType` that can be allocated.
         */
        static sizeType maxSize(const Alloc& alloc) {
            if constexpr (hasMaxSize::value) {
                return alloc.maxSize();
            }
            else {
                return std::numeric_limits<sizeType>::max() / sizeof(valueType);
            }
        }
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_ALLOCATORTRAITS_CUH
