cmake_minimum_required(VERSION 3.25)
project(luca-vinciguerra-bathesis)

set(CMAKE_CXX_STANDARD 17)

configure_file(resources/datasets/magic_normalized.txt magic_normalized.txt COPYONLY)
configure_file(scripts/benchmark.bat benchmark.bat COPYONLY)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(lucavinciguerra-bathesis src/main.cpp
        src/npp/npp.h
        src/npp/npp.cpp
        src/threadpool/threadpool.h)
