cmake_minimum_required(VERSION 3.15)
project(mini-llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(test_tensor 
    src/test_tensor.cpp
    src/tensor.cpp
)

# Automatically add all .cpp files in src
# file(GLOB SOURCES "src/*.cpp")
# add_executable(test_tensor ${SOURCES})