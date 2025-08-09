cmake_minimum_required(VERSION 3.18)
project(CuWeaver VERSION 0.1.0 LANGUAGES CUDA CXX)

option(CUWEAVER_ENABLE_WARNINGS "Enable extra warnings" OFF)
option(CUWEAVER_USE_CUDA_RUNTIME "Link against CUDA Runtime (cudart). OFF means Driver API only" ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(CUWEAVER_ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4>" "$<$<COMPILE_LANGUAGE:CUDA>:/W3>")
    else()
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
    endif()
endif()

if(NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>")
endif()

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(src)


add_subdirectory(examples)



include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()


include(CMakePackageConfigHelpers)

install(EXPORT CuWeaverTargets
        NAMESPACE cuweaver::
        FILE CuWeaverTargets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CuWeaver"
)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CuWeaverConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CuWeaverConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CuWeaver"
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CuWeaverConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/CuWeaverConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/CuWeaverConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CuWeaver"
)