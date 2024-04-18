cmake_minimum_required(VERSION 3.28)
project(code)

set(CMAKE_CXX_STANDARD 11)

add_executable(code main.cpp
        npp/npp.h
        npp/npp.cpp)
