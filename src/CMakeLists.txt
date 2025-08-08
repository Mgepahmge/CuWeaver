set(CUWEAVER_PUBLIC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(CUWEAVER_API_DIR "${CUWEAVER_PUBLIC_INCLUDE_DIR}/cuweaver")
set(CUWEAVER_UTILS_DIR "${CUWEAVER_PUBLIC_INCLUDE_DIR}/cuweaver_utils")

file(GLOB CUWEAVER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(cuweaver STATIC ${CUWEAVER_SOURCES})
add_library(cuweaver::cuweaver ALIAS cuweaver)

target_include_directories(cuweaver
        PUBLIC
        "$<BUILD_INTERFACE:${CUWEAVER_PUBLIC_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

find_package(CUDAToolkit REQUIRED)

if(CUWEAVER_USE_CUDA_RUNTIME)
    target_link_libraries(cuweaver PUBLIC CUDA::cudart_static)
    target_compile_definitions(cuweaver PUBLIC CUWEAVER_USE_CUDA_RUNTIME=1)
else()
    target_link_libraries(cuweaver PUBLIC CUDA::cuda_driver)
    target_compile_definitions(cuweaver PUBLIC CUWEAVER_USE_CUDA_RUNTIME=0)
endif()

target_compile_definitions(cuweaver
        PUBLIC
        CUWEAVER_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CUWEAVER_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CUWEAVER_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

install(TARGETS cuweaver
        EXPORT CuWeaverTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(DIRECTORY "${CUWEAVER_API_DIR}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cuweaver"
        FILES_MATCHING PATTERN "*.cuh"
)

install(DIRECTORY "${CUWEAVER_UTILS_DIR}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cuweaver_utils"
        FILES_MATCHING PATTERN "*.cuh"
)