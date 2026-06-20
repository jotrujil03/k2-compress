cmake_minimum_required(VERSION 3.20)
project(k2 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

set(K2_ROOT        ${CMAKE_CURRENT_SOURCE_DIR})
set(OPENZL_ROOT    ${K2_ROOT}/../openzl)
set(K2_PY_DIR      ${K2_ROOT}/src/python)

# ---------------------------------------------------------------------------
# OpenZL
# ---------------------------------------------------------------------------

# if(NOT EXISTS ${OPENZL_ROOT}/CMakeLists.txt)
#    message(FATAL_ERROR "OpenZL not found at ${OPENZL_ROOT}. "
#            "Clone it with: git clone --depth 1 -b release "
#            "https://github.com/facebook/openzl.git ../openzl")
# endif()

# Build OpenZL as a subdirectory (gives us the zl target)
# set(OPENZL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# set(OPENZL_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
# add_subdirectory(${OPENZL_ROOT} openzl_build)

# ---------------------------------------------------------------------------
# ASDP-LH (Replaces OpenZL)
# ---------------------------------------------------------------------------

set(ASDP_ROOT ${K2_ROOT}/../asdp-lh)
if(NOT EXISTS ${ASDP_ROOT}/CMakeLists.txt)
    message(FATAL_ERROR "ASDP-LH not found at ${ASDP_ROOT}.")
endif()

# Build ASDP-LH as a subdirectory
set(ASDP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${ASDP_ROOT} asdp_build)

# ---------------------------------------------------------------------------
# Python 3.14
# ---------------------------------------------------------------------------

find_package(Python3 3.14 EXACT COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python3: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libs: ${Python3_LIBRARIES}")

# ---------------------------------------------------------------------------
# pybind11
# ---------------------------------------------------------------------------

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()

list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

message(STATUS "pybind11: ${pybind11_VERSION} at ${PYBIND11_CMAKE_DIR}")

# ---------------------------------------------------------------------------
# K2 bridge shared library
# ---------------------------------------------------------------------------
# Produces: libk2.so (Linux) / libk2.dylib (macOS)
# Defines K2_IMPLEMENTATION so the C API bodies are compiled in.

add_library(k2 SHARED
    src/cpp/k2_bridge.cpp
)

set_target_properties(k2 PROPERTIES
    C_VISIBILITY_PRESET   hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(k2 PRIVATE
    K2_PYTHON_MODULE_DIR="${K2_PY_DIR}"
)

target_include_directories(k2
    PUBLIC
        $<BUILD_INTERFACE:${K2_ROOT}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
        # ${OPENZL_ROOT}/cpp/include
        # ${OPENZL_ROOT}/include
        ${ASDP_ROOT}/include
)

target_link_libraries(k2
    PRIVATE
        pybind11::embed
        # openzl
        # openzl_cpp
        asdp
        ${Python3_LIBRARIES}
)

# ---------------------------------------------------------------------------
# K2 CLI tool (optional — compress/decompress from command line)
# ---------------------------------------------------------------------------

add_executable(k2cli
    src/cpp/k2cli.cpp
    src/cpp/k2archive.cpp
)

target_include_directories(k2cli PRIVATE
    ${K2_ROOT}/include
    ${ASDP_ROOT}/include
)
target_link_libraries(k2cli 
    PRIVATE 
        k2 
        # openzl 
        # openzl_cpp 
        asdp
        pybind11::embed
)
target_compile_definitions(k2cli PRIVATE K2_PYTHON_MODULE_DIR="${K2_PY_DIR}")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

option(K2_BUILD_TESTS "Build C++ tests" ON)

if(K2_BUILD_TESTS)
    enable_testing()

    add_executable(k2_test
        src/cpp/k2_test.cpp
    )

    target_include_directories(k2_test PRIVATE
        ${K2_ROOT}/include
        ${ASDP_ROOT}/include
    )
    target_link_libraries(k2_test 
        PRIVATE 
            k2 
            # openzl 
            # openzl_cpp 
            asdp
            pybind11::embed
    )
    target_compile_definitions(k2_test PRIVATE K2_PYTHON_MODULE_DIR="${K2_PY_DIR}")

    add_test(NAME k2_bridge_test COMMAND k2_test)
    set_tests_properties(k2_bridge_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${Python3_SITELIB}"
    )

    # k2archive has no pybind11/Python dependency — pure C++ on top of asdp.
    add_executable(k2archive_test
        tests/k2archive_test.cpp
        src/cpp/k2archive.cpp
    )
    target_include_directories(k2archive_test PRIVATE
        ${K2_ROOT}/include
        ${ASDP_ROOT}/include
    )
    target_link_libraries(k2archive_test PRIVATE asdp)
    add_test(NAME k2archive_test COMMAND k2archive_test)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS k2 k2cli
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin
)

install(FILES include/k2_bridge.h DESTINATION include)

install(DIRECTORY src/python/
    DESTINATION lib/k2/python
    FILES_MATCHING PATTERN "*.py"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "K2 configuration:")
message(STATUS "  Version      : ${PROJECT_VERSION}")
message(STATUS "  ASDP root    : ${ASDP_ROOT}")
message(STATUS "  Python dir   : ${K2_PY_DIR}")
message(STATUS "  Python       : ${Python3_VERSION} (${Python3_EXECUTABLE})")
message(STATUS "  pybind11     : ${pybind11_VERSION}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests  : ${K2_BUILD_TESTS}")
message(STATUS "")
