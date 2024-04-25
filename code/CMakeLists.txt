cmake_minimum_required(VERSION 3.25)
project(luca-vinciguerra-bathesis)

set(CMAKE_CXX_STANDARD 17)

configure_file(resources/magic_normalized.txt magic_normalized.txt COPYONLY)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(lucavinciguerra-bathesis main.cpp
        npp/npp.h
        npp/npp.cpp
        threadpool/threadpool.h)
