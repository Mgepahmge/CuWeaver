#include <cuweaver_utils/Error.cuh>

namespace cuweaver {
    const char* cudaErrorCategoryImpl::name() const noexcept {
        return "cuda";
    }

    std::string cudaErrorCategoryImpl::message(int ev) const {
        auto e = static_cast<cudaError_t>(ev);
        if (const char* s = cudaGetErrorString(e)) {
            return s;
        }
        return "Unknown CUDA error";
    }

    cudaError::cudaError(cudaError_t e, cudaErrorContext ctx) : std::system_error(makeErrorCode(e),
                                                                    buildWhat(ctx)),
                                                                code_(e),
                                                                ctx_(std::move(ctx)) {
    }

    cudaError_t cudaError::codeNative() const noexcept {
        return code_;
    }

    const cudaErrorContext& cudaError::context() const noexcept {
        return ctx_;
    }

    std::string cudaError::buildWhat(const cudaErrorContext& c) {
        std::string s = c.op;
        if (!c.detail.empty()) {
            s += " | ";
            s += c.detail;
        }
        s += " @ ";
        s += c.loc.file;
        s += ":";
        s += std::to_string(c.loc.line);
        return s;
    }
}
