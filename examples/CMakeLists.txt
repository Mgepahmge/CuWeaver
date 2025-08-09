file(GLOB EXAMPLE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    target_link_libraries(${EXAMPLE_NAME}
            PRIVATE
            cuweaver::cuweaver
    )

    set_target_properties(${EXAMPLE_NAME} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
    )

    set_target_properties(${EXAMPLE_NAME} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
    )

    if(NOT WIN32)
        set_target_properties(${EXAMPLE_NAME} PROPERTIES
                BUILD_RPATH_USE_ORIGIN TRUE
                INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endforeach()

add_custom_target(examples
        DEPENDS ${EXAMPLE_TARGETS}
        COMMENT "Building all examples"
)