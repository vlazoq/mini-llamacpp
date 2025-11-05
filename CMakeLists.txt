cmake_minimum_required(VERSION 3.15)
project(mini-llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
include_directories(external)  # For doctest.h and gtest

# Source files (our library)
set(SOURCES
    src/tensor.cpp
)


# ===== DOCTEST TESTS =====
add_executable(test_doctest 
    ${SOURCES}
    tests/doctest/test_tensor.cpp
)


# ===== GOOGLE TEST SETUP =====
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Test executable
add_executable(test_gtest
    ${SOURCES}
    tests/gtest/test_tensor.cpp
)
target_link_libraries(test_gtest GTest::gtest_main)



# Enable testing
enable_testing()


# Register both test suites
add_test(NAME DoctestSuite COMMAND test_doctest)
add_test(NAME GtestSuite COMMAND test_gtest)

# Main executable (add later)
# add_executable(mini-llama ${SOURCES} src/main.cpp)

# Phase 1 Demo
# add_executable(phase1_demo
#    ${SOURCES}
#    src/phase1_demo.cpp)