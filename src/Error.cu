#include <cuweaver_utils/Error.cuh>

namespace cuweaver {
    const char* cuda_error_category_impl::name() const noexcept {
        return "cuda";
    }

    std::string cuda_error_category_impl::message(int ev) const {
        auto e = static_cast<cudaError_t>(ev);
        if (const char* s = cudaGetErrorString(e)) {
            return s;
        }
        return "Unknown CUDA error";
    }

    cuda_error::cuda_error(cudaError_t e, cuda_error_context ctx) : std::system_error(make_error_code(e),
                                                                        build_what(ctx)),
                                                                    code_(e),
                                                                    ctx_(std::move(ctx)) {
    }

    cudaError_t cuda_error::code_native() const noexcept {
        return code_;
    }

    const cuda_error_context& cuda_error::context() const noexcept {
        return ctx_;
    }

    std::string cuda_error::build_what(const cuda_error_context& c) {
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
