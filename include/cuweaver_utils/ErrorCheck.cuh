/**
 * @file ErrorCheck.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Provides CUDA error checking utilities and macros for context-aware error handling.
 *
 * @details This header defines CUDA-specific error handling utilities within the `cuweaver` namespace, including:
 * - Overloaded `throwIfError` functions that throw `cudaError` exceptions with contextual metadata (operation name,
 *   optional details, and source location) when a CUDA error (`cudaError_t`) occurs.
 * - A `check` function that converts a `cudaError_t` to a `std::error_code` for non-throwing error propagation.
 * - Convenience macros:
 *   - `CUW_HERE`: Captures the current source location (file, line, function) using standard preprocessor macros.
 *   - `CUW_THROW_IF_ERROR`: Wraps CUDA API calls to automatically check results and throw context-rich errors on failure.
 *
 * This header depends on `Error.cuh` for core error types (`cudaError`, `cudaErrorContext`, `SourceLocation`).
 */
#ifndef CUWEAVER_ERRORCHECK_CUH
#define CUWEAVER_ERRORCHECK_CUH

#ifdef __CUDACC__

#include "Error.cuh"
#include <string_view>

namespace cuweaver {
    /**
     * @brief Throws a cudaError with context if a CUDA error occurred.
     *
     * @details Checks if the provided CUDA status is not cudaSuccess. If so, constructs a cudaErrorContext with
     *          the operation name, detail message, and source location, then throws a cudaError containing this context.
     *
     * @param[in] st CUDA error status to check.
     * @param[in] op Name of the CUDA operation that failed.
     * @param[in] detail Additional detail about the failure context.
     * @param[in] loc Source location where the error check occurred.
     *
     * @throws cudaError Thrown if st is not cudaSuccess.
     */
    inline void throwIfError(const cudaError_t st,
                             const std::string_view op,
                             const std::string_view detail,
                             const SourceLocation& loc) {
        if (st != cudaSuccess) {
            throw cudaError(st, cudaErrorContext{std::string(op), std::string(detail), loc});
        }
    }

    /**
     * @brief Throws a cudaError with context (without detail) if a CUDA error occurred.
     *
     * @details Overload of throwIfError that omits the detail message. Calls the full overload with an empty detail string.
     *
     * @param[in] st CUDA error status to check.
     * @param[in] op Name of the CUDA operation that failed.
     * @param[in] loc Source location where the error check occurred.
     *
     * @throws cudaError Thrown if st is not cudaSuccess.
     */
    inline void throwIfError(const cudaError_t st,
                             const std::string_view op,
                             const SourceLocation& loc) {
        throwIfError(st, op, {}, loc);
    }

    /**
    * @brief Converts a CUDA error status to a std::error_code.
    *
    * @details Uses makeErrorCode to create a std::error_code representing the CUDA error status. This function is noexcept.
    *
    * @param[in] st CUDA error status to convert.
    *
    * @return std::error_code corresponding to the CUDA error status.
    */
    inline std::error_code check(const cudaError_t st) noexcept {
        return makeErrorCode(st);
    }

    /**
     * @def CUW_HERE
     * @brief Captures the current source location (file, line, function).
     *
     * @details Uses standard preprocessor macros __FILE__, __LINE__, and __func__ to create a SourceLocation object
     *          representing the current position in the code.
     */
#define CUW_HERE ::cuweaver::here(__FILE__, __LINE__, __func__)

    /**
     * @def CUW_THROW_IF_ERROR
     * @brief Checks a CUDA operation's result and throws a contextual error if it failed.
     *
     * @details Wraps a CUDA API call (expr) to check its return status. If the status is not cudaSuccess, calls throwIfError
     *          with the operation name (stringified expr), optional detail arguments, and the current source location (via CUW_HERE).
     *
     * @param expr CUDA API call to execute and check.
     * @param ... Optional detail string to include in the error context.
     */
#define CUW_THROW_IF_ERROR(expr, ...) \
::cuweaver::throwIfError((expr), #expr, ##__VA_ARGS__, CUW_HERE)
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_ERRORCHECK_CUH
