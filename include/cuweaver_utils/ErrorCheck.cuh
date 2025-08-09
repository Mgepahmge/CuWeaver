#ifndef CUWEAVER_ERRORCHECK_CUH
#define CUWEAVER_ERRORCHECK_CUH

#include "Error.cuh"
#include <string_view>

namespace cuweaver {
    inline void throw_if_error(const cudaError_t st,
                               const std::string_view op,
                               const std::string_view detail,
                               const source_location& loc) {
        if (st != cudaSuccess) {
            throw cuda_error(st, cuda_error_context{std::string(op), std::string(detail), loc});
        }
    }

    inline void throw_if_error(const cudaError_t st,
                               const std::string_view op,
                               const source_location& loc) {
        throw_if_error(st, op, {}, loc);
    }

    inline std::error_code check(const cudaError_t st) noexcept {
        return make_error_code(st);
    }


#define CUW_HERE ::cuweaver::here(__FILE__, __LINE__, __func__)

#define CUW_THROW_IF_ERROR(expr, ...) \
::cuweaver::throw_if_error((expr), #expr, ##__VA_ARGS__, CUW_HERE)
}

#endif //CUWEAVER_ERRORCHECK_CUH
