#include <gtest/gtest.h>

#include <system_error>
#include <string>

#include "cuweaver_utils/ErrorCheck.cuh"

namespace {

static bool contains_substr(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

static void invoke_throw_if_error(cudaError_t code) {
    CUW_THROW_IF_ERROR(code);
}

static void invoke_throw_if_error_with_detail(cudaError_t code, const char* detail) {
    CUW_THROW_IF_ERROR(code, detail);
}

TEST(CudaRuntimeTest, MallocAndFreeSuccess) {
    void* d_ptr = nullptr;
    EXPECT_NO_THROW({
        CUW_THROW_IF_ERROR(cudaMalloc(&d_ptr, 1024));
        CUW_THROW_IF_ERROR(cudaFree(d_ptr));
    });
}

TEST(CudaRuntimeTest, MallocFailure) {
    void* d_ptr = nullptr;
    const size_t big = static_cast<size_t>(1ULL << 40);
    try {
        CUW_THROW_IF_ERROR(cudaMalloc(&d_ptr, big));
        cudaFree(d_ptr);
        FAIL() << "Expected cuweaver::cuda_error to be thrown";
    } catch (const cuweaver::cuda_error& ex) {
        EXPECT_NE(ex.code_native(), cudaSuccess);
        const std::string expected_op = "cudaMalloc(&d_ptr, big)";
        EXPECT_EQ(ex.context().op, expected_op);
        EXPECT_TRUE(ex.context().detail.empty());
        EXPECT_TRUE(contains_substr(ex.context().loc.file, "ErrorHandlingTest.cu"));
    } catch (...) {
        FAIL() << "Unexpected exception type";
    }
}

TEST(CudaRuntimeTest, MemsetInvalidValue) {
    void* d_ptr = nullptr;
    const size_t count = 4;
    try {
        CUW_THROW_IF_ERROR(cudaMemset(d_ptr, 0, count));
        FAIL() << "Expected cuweaver::cuda_error to be thrown";
    } catch (const cuweaver::cuda_error& ex) {
        EXPECT_NE(ex.code_native(), cudaSuccess);
        EXPECT_EQ(ex.context().op, std::string("cudaMemset(d_ptr, 0, count)"));
        EXPECT_TRUE(ex.context().detail.empty());
    } catch (...) {
        FAIL() << "Unexpected exception type";
    }
}

TEST(CudaErrorHandlingTest, DoesNotThrowOnSuccess) {
    EXPECT_NO_THROW({
        invoke_throw_if_error(cudaSuccess);
    });
}

TEST(CudaErrorHandlingTest, ThrowsWithProperContext) {
    const cudaError_t code = cudaErrorInvalidValue;
    try {
        invoke_throw_if_error(code);
        FAIL() << "Expected cuweaver::cuda_error to be thrown";
    } catch (const cuweaver::cuda_error& ex) {
        EXPECT_EQ(ex.code_native(), code);
        const std::error_code ec = ex.code();
        EXPECT_EQ(ec.value(), static_cast<int>(code));
        EXPECT_STREQ(ec.category().name(), cuweaver::cuda_error_category().name());
        EXPECT_EQ(ex.context().op, std::string("code"));
        EXPECT_TRUE(ex.context().detail.empty());
        EXPECT_TRUE(contains_substr(ex.context().loc.file, "ErrorHandlingTest.cu"));
        EXPECT_TRUE(contains_substr(ex.context().loc.function, "invoke_throw_if_error"));
        EXPECT_GT(ex.context().loc.line, 0);
    } catch (...) {
        FAIL() << "Unexpected exception type";
    }
}


TEST(CudaErrorHandlingTest, ThrowsWithCustomDetail) {
    const cudaError_t code = cudaErrorMemoryAllocation;
    const char* detail = "Allocation of temporary buffer failed";
    try {
        invoke_throw_if_error_with_detail(code, detail);
        FAIL() << "Expected cuweaver::cuda_error to be thrown";
    } catch (const cuweaver::cuda_error& ex) {
        EXPECT_EQ(ex.code_native(), code);
        const auto& ctx = ex.context();
        EXPECT_EQ(ctx.op, std::string("code"));
        EXPECT_EQ(ctx.detail, std::string(detail));
        EXPECT_TRUE(contains_substr(ctx.loc.file, "ErrorHandlingTest.cu"));
        EXPECT_TRUE(contains_substr(ctx.loc.function, "invoke_throw_if_error_with_detail"));
    } catch (...) {
        FAIL() << "Unexpected exception type";
    }
}

TEST(CudaErrorHandlingTest, ThrowIfErrorOverloadWithCustomOpAndDetail) {
    const cudaError_t code = cudaErrorInvalidDevice;
    const auto loc = CUW_HERE;
    const std::string op = "Launching kernel";
    const std::string detail = "Device ID 99 does not exist";
    try {
        cuweaver::throw_if_error(code, op, detail, loc);
        FAIL() << "Expected cuweaver::cuda_error to be thrown";
    } catch (const cuweaver::cuda_error& ex) {
        EXPECT_EQ(ex.code_native(), code);
        const auto& ctx = ex.context();
        EXPECT_EQ(ctx.op, op);
        EXPECT_EQ(ctx.detail, detail);
        EXPECT_EQ(ctx.loc.file, loc.file);
        EXPECT_EQ(ctx.loc.line, loc.line);
        EXPECT_STREQ(ctx.loc.function, loc.function);
    } catch (...) {
        FAIL() << "Unexpected exception type";
    }
}

TEST(CudaErrorHandlingTest, ThrowIfErrorOverloadWithoutDetail) {
    const cudaError_t code = cudaErrorNotSupported;
    const auto loc = CUW_HERE;
    const std::string op = "Unsupported operation";
    try {
        cuweaver::throw_if_error(code, op, loc);
        FAIL() << "Expected cuweaver::cuda_error to be thrown";
    } catch (const cuweaver::cuda_error& ex) {
        EXPECT_EQ(ex.code_native(), code);
        const auto& ctx = ex.context();
        EXPECT_EQ(ctx.op, op);
        EXPECT_TRUE(ctx.detail.empty());
        EXPECT_EQ(ctx.loc.file, loc.file);
        EXPECT_EQ(ctx.loc.line, loc.line);
        EXPECT_STREQ(ctx.loc.function, loc.function);
    } catch (...) {
        FAIL() << "Unexpected exception type";
    }
}

TEST(CudaErrorHandlingTest, CheckReturnsExpectedErrorCode) {
    {
        std::error_code ec = cuweaver::check(cudaSuccess);
        EXPECT_EQ(ec.value(), 0);
        EXPECT_FALSE(ec);
        EXPECT_STREQ(ec.category().name(), cuweaver::cuda_error_category().name());
    }
    {
        const cudaError_t code = cudaErrorInvalidPitchValue;
        std::error_code ec = cuweaver::check(code);
        EXPECT_EQ(ec.value(), static_cast<int>(code));
        EXPECT_TRUE(ec);
        EXPECT_STREQ(ec.category().name(), cuweaver::cuda_error_category().name());
    }
}

TEST(CudaErrorHandlingTest, CudaErrorCategoryIsSingleton) {
    const std::error_category& cat1 = cuweaver::cuda_error_category();
    const std::error_category& cat2 = cuweaver::cuda_error_category();
    EXPECT_EQ(&cat1, &cat2);
    EXPECT_FALSE(std::string(cat1.name()).empty());
}

} // namespace