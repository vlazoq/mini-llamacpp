cmake_minimum_required(VERSION 3.15)
project(mini-llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
include_directories(external)  # For doctest.h

# Source files (our library)
set(SOURCES
    src/tensor.cpp
)

# Main executable (add later)
# add_executable(mini-llama ${SOURCES} src/main.cpp)


# Automatically add all .cpp files in src
# file(GLOB SOURCES "src/*.cpp")
# add_executable(test_tensor ${SOURCES})


# Test executable
add_executable(test_tensor
    ${SOURCES}
    tests/test_tensor.cpp
)

# Enable testing
enable_testing()
add_test(NAME TensorTests COMMAND test_tensor)