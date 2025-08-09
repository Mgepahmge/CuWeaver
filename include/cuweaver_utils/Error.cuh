#ifndef CUWEAVER_ERROR_CUH
#define CUWEAVER_ERROR_CUH

#include <system_error>
#include <string>

namespace cuweaver {
    struct source_location {
        const char* file = "unknown";
        int line = 0;
        const char* function = "unknown";
    };

    constexpr source_location here(const char* file, int line, const char* func) noexcept {
        return {file, line, func};
    }

    class cuda_error_category_impl final : public std::error_category {
    public:
        [[nodiscard]] const char* name() const noexcept override;

        [[nodiscard]] std::string message(int ev) const override;
    };

    inline const std::error_category& cuda_error_category() {
        static cuda_error_category_impl cat;
        return cat;
    }

    inline std::error_code make_error_code(cudaError_t e) noexcept {
        return {static_cast<int>(e), cuda_error_category()};
    }

    struct cuda_error_context {
        std::string op;
        std::string detail;
        source_location loc{};
    };

    class cuda_error : public std::system_error {
    public:
        cuda_error(cudaError_t e, cuda_error_context ctx);

        [[nodiscard]] cudaError_t code_native() const noexcept;

        [[nodiscard]] const cuda_error_context& context() const noexcept;

    private:
        static std::string build_what(const cuda_error_context& c);

        cudaError_t code_;
        cuda_error_context ctx_;
    };
} // namespace cuweaver

namespace std {
    template <>
    struct is_error_code_enum<cudaError_t> : true_type {
    };
}

#endif //CUWEAVER_ERROR_CUH
