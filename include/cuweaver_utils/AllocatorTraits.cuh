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
         * @brief SFINAE utility alias that maps any type pack to `void`.
         *
         * @details Converts an arbitrary list of types into `void`, enabling Substitution Failure Is Not An Error (SFINAE)
         *          in template metaprogramming. Used to detect whether a given expression or template instantiation is valid
         *          by forcing substitution failures to exclude invalid template branches.
         * @tparam ... Unused type parameters; all input types are ignored.
         */
        template <typename...>
        using void_t = void;

        /**
         * @struct detector
         * @brief Primary template for the detector pattern, a metafunction to check if a template operation is valid.
         *
         * @details Serves as the base case for the detector pattern. Returns `std::false_type` (via `value_t`) to indicate
         *          the operation is invalid for the given arguments, and uses `Default` as the fallback result type.
         *
         * @tparam Default Type returned as `type` when the operation is invalid.
         * @tparam AlwaysVoid Placeholder type for SFINAE; typically `void_t<Op<Args...>>` to trigger substitution checks.
         * @tparam Op Template template parameter representing the operation to validate (e.g., a trait or member function).
         * @tparam Args Type arguments to pass to the operation `Op`.
         */
        template <typename Default, typename AlwaysVoid, template<typename...> typename Op, typename... Args>
        struct detector {
            using value_t = std::false_type;
            using type = Default;
        };

        /**
         * @struct detector
         * @brief Partial specialization of `detector` for valid operations.
         *
         * @details Activates when the template operation `Op<Args...>` is valid (i.e., `void_t<Op<Args...>>` resolves to `void`).
         *          Returns `std::true_type` (via `value_t`) to indicate validity and sets `type` to the result of `Op<Args...>`.
         *
         * @tparam Default Unused fallback type (overridden by the valid operation result).
         * @tparam Op Template template parameter representing the valid operation.
         * @tparam Args Type arguments for which `Op<Args...>` is a valid instantiation.
         */
        template <typename Default, template<typename...> typename Op, typename... Args>
        struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
            using value_t = std::true_type;
            using type = Op<Args...>;
        };
    }

    /**
     * @brief Type alias for the detector's boolean result type, indicating if a template operation is valid.
     *
     * @details A shorthand for `detail::detector<void, void, Op, Args...>::value_t`, which resolves to
     *          `std::true_type` if the template operation `Op<Args...>` is a valid instantiation, or
     *          `std::false_type` otherwise. Uses the detector pattern to perform SFINAE-based validity checks.
     *
     * @tparam Op Template template parameter representing the operation to validate (e.g., a trait or member function template).
     * @tparam Args Type arguments to pass to the operation `Op`.
     */
    template <template<typename...> typename Op, typename... Args>
    using is_detected = typename detail::detector<void, void, Op, Args...>::value_t;

    /**
     * @brief Constexpr boolean value indicating if a template operation is valid.
     *
     * @details A constexpr alias for `is_detected<Op, Args...>::value`, providing a direct boolean result
     *          of whether the template operation `Op<Args...>` is a valid instantiation.
     *
     * @tparam Op Template template parameter representing the operation to validate.
     * @tparam Args Type arguments to pass to the operation `Op`.
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
         * @brief Decltype alias for the allocator's `construct` member function (if present).
         *
         * @tparam T Type of the object to construct.
         * @tparam Args Types of arguments to pass to the object's constructor.
         */
        template <typename T, typename... Args>
        using construct_t = decltype(std::declval<Alloc&>().construct(std::declval<T*>(), std::declval<Args>()...));

        /**
         * @brief Type trait indicating if the allocator has a valid `construct` member function for the given types.
         *
         * @tparam T Type of the object to construct.
         * @tparam Args Types of arguments for the constructor.
         */
        template <typename T, typename... Args>
        using hasConstruct = is_detected<construct_t, T, Args...>;

        /**
         * @brief Decltype alias for the allocator's `destroy` member function (if present).
         *
         * @tparam T Type of the object to destroy.
         */
        template <typename T>
        using destroy_t = decltype(std::declval<Alloc&>().destroy(std::declval<T*>()));

        /**
         * @brief Type trait indicating if the allocator has a valid `destroy` member function for the given type.
         *
         * @tparam T Type of the object to destroy.
         */
        template <typename T>
        using hasDestroy = is_detected<destroy_t, T>;

        /**
         * @brief Decltype alias for the allocator's `memcpy` member function (if present).
         *
         * @tparam T Type of the pointers involved in the memory copy.
         */
        template <typename T>
        using memcpy_t = decltype(std::declval<Alloc&>().memcpy(std::declval<T*>(), std::declval<T*>(),
                                                                std::declval<sizeType>(),
                                                                std::declval<cudaMemcpyKind>()));

        /**
         * @brief Type trait indicating if the allocator has a valid `memcpy` member function for the given type.
         *
         * @tparam T Type of the pointers involved in the memory copy.
         */
        template <typename T>
        using hasMemcpy = is_detected<memcpy_t, T>;

        using maxSize_t = decltype(std::declval<const Alloc&>().maxSize());
        //!< Decltype alias for the allocator's `maxSize` member function (if present).
        using hasMaxSize = is_detected<maxSize_t>;
        //!< Type trait indicating if the allocator has a valid `maxSize` member function.

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
        template <typename T, typename... Args>
        static void destroy(Alloc& alloc, T* p) {
            if constexpr (hasDestroy<T>::value) {
                alloc.destroy(p);
            }
            else {
                p->~T();
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
#pragma warning("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_ALLOCATORTRAITS_CUH
