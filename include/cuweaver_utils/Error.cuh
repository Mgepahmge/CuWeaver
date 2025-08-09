/**
 * @file Error.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Provides CUDA-specific error handling utilities and types.
 *
 * @details Defines components for consistent CUDA error reporting and propagation, including:
 * - A `SourceLocation` struct to capture error origin information (file, line, function).
 * - A custom `std::error_category` implementation (`cudaErrorCategoryImpl`) for translating CUDA error codes to human-readable messages.
 * - A `cudaError` exception type that extends `std::system_error` with additional context (operation description, source location).
 * - Helper functions like `makeErrorCode` to integrate CUDA errors with C++ standard error handling workflows.
 * - A specialization of `std::is_error_code_enum` for `cudaError_t` to enable implicit conversion to `std::error_code`.
 *
 * These utilities enable type-safe, context-rich error handling for CUDA operations, aligning with C++ standard library patterns while preserving CUDA-specific error semantics.
 */

#ifndef CUWEAVER_ERROR_CUH
#define CUWEAVER_ERROR_CUH

#include <system_error>
#include <string>

namespace cuweaver {
    /**
     * @struct SourceLocation
     * @brief Represents a source code location (file, line, and function).
     *
     * @details Used to capture contextual information about where an error occurred
     *          in the source code for debugging and error reporting.
     */
    struct SourceLocation {
        const char* file = "unknown"; //!< Name of the source file where the error occurred.
        int line = 0; //!< Line number in the source file where the error occurred.
        const char* function = "unknown"; //!< Name of the function where the error occurred.
    };

    /**
     * @brief Creates a SourceLocation instance with specified file, line, and function details.
     *
     * @param[in] file Path to the source file (usually set via __FILE__).
     * @param[in] line Line number in the source file (usually set via __LINE__).
     * @param[in] func Name of the function (usually set via __func__ or __PRETTY_FUNCTION__).
     * @return A SourceLocation instance initialized with the provided location details.
     */
    constexpr SourceLocation here(const char* file, int line, const char* func) noexcept {
        return {file, line, func};
    }

    /**
     * @class cudaErrorCategoryImpl
     * @brief Implementation of std::error_category for CUDA runtime errors.
     *
     * @details Provides CUDA-specific error message translation and category identification
     *          by overriding std::error_category's virtual methods.
     */
    class cudaErrorCategoryImpl final : public std::error_category {
    public:
        /**
         * @brief Returns the name of the CUDA error category.
         * @return A null-terminated string representing the category name ("cuda").
         */
        [[nodiscard]] const char* name() const noexcept override;

        /**
         * @brief Returns a human-readable message for a given CUDA error code.
         * @param[in] ev Integer representation of a cudaError_t value.
         * @return A string describing the CUDA error corresponding to @p ev.
         */
        [[nodiscard]] std::string message(int ev) const override;
    };

    /**
     * @brief Retrieves the singleton instance of the CUDA error category.
     * @return A constant reference to the singleton cudaErrorCategoryImpl instance.
     */
    inline const std::error_category& cudaErrorCategory() {
        static cudaErrorCategoryImpl cat;
        return cat;
    }

    /**
     * @brief Creates a std::error_code instance from a CUDA error code.
     *
     * @param[in] e CUDA error code to convert to a std::error_code.
     * @return std::error_code representing the CUDA error, using the CUDA error category.
     */
    inline std::error_code makeErrorCode(cudaError_t e) noexcept {
        return {static_cast<int>(e), cudaErrorCategory()};
    }

    /**
     * @struct cudaErrorContext
     * @brief Holds contextual metadata for a CUDA error (e.g., failing operation, source location).
     */
    struct cudaErrorContext {
        std::string op; //!< Description of the operation that failed (e.g., "cudaMalloc")
        std::string detail; //!< Additional details about the error (e.g., parameter values).
        SourceLocation loc{}; //!< Source code location where the error occurred.
    };

    /**
     * @class cudaError
     * @brief Exception type wrapping CUDA errors with additional contextual information.
     *
     * @details Extends std::system_error to include CUDA-specific error codes and context (operation,
     *          details, source location) for more informative error reporting.
     */
    class cudaError : public std::system_error {
    public:
        /**
         * @brief Constructs a cudaError with a CUDA error code and contextual information.
         *
         * @param[in] e CUDA error code representing the failure.
         * @param[in] ctx Contextual metadata (operation, details, source location) about the error.
         */
        cudaError(cudaError_t e, cudaErrorContext ctx);

        /**
         * @brief Gets the original CUDA error code that triggered the exception.
         *
         * @return Original CUDA error code (cudaError_t) passed to the constructor.
         */
        [[nodiscard]] cudaError_t codeNative() const noexcept;

        /**
         * @brief Gets the immutable contextual metadata associated with the error.
         *
         * @return Constant reference to the cudaErrorContext containing operation details,
         *         additional information, and source location.
         */
        [[nodiscard]] const cudaErrorContext& context() const noexcept;

    private:
        /**
         * @brief Builds the formatted what() string for the exception using context data.
         *
         * @param[in] c Contextual metadata to include in the what() message.
         * @return Constructed string with error context (operation, details, source location).
         */
        static std::string buildWhat(const cudaErrorContext& c);

        cudaError_t code_; //!< Original CUDA error code stored for the exception.
        cudaErrorContext ctx_; //!< Contextual metadata for the error (operation, details, location).
    };
} // namespace cuweaver

namespace std {
    /**
     * @struct std::is_error_code_enum<cudaError_t>
     * @brief Specializes std::is_error_code_enum to recognize cudaError_t as an error code enum.
     *
     * @details This template specialization marks cudaError_t as compatible with C++ standard
     *          library error handling. It enables implicit conversion of cudaError_t values to
     *          std::error_code, using the CUDA error category for message translation.
     */
    template <>
    struct is_error_code_enum<cudaError_t> : true_type {
    };
}

#endif //CUWEAVER_ERROR_CUH
