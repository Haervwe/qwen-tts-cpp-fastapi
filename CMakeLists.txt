cmake_minimum_required(VERSION 3.14)
project(qwen-tts-api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add the submodule
add_subdirectory(qwen3-tts.cpp)
