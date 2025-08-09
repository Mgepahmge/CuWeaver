find_package(GTest REQUIRED)

file(GLOB TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    set(TEST_TARGET "test_${TEST_NAME}")

    add_executable(${TEST_TARGET} ${TEST_SOURCE})

    target_link_libraries(${TEST_TARGET}
            PRIVATE
            cuweaver::cuweaver
            GTest::GTest
            GTest::Main
    )

    set_target_properties(${TEST_TARGET} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
    )

    set_target_properties(${TEST_TARGET} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
    )

    if(NOT WIN32)
        set_target_properties(${TEST_TARGET} PROPERTIES
                BUILD_RPATH_USE_ORIGIN TRUE
                INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})

    set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 300
    )
endforeach()

add_custom_target(build_tests
        DEPENDS ${TEST_TARGETS}
        COMMENT "Building all tests"
)