cmake_minimum_required(VERSION 3.28)
project(code)

set(CMAKE_CXX_STANDARD 17)

#configure_file(resources/winequality_white_normalized.txt winequality_white_normalized.txt COPYONLY)
#configure_file(resources/startnet_seed_1.net startnet_seed_1.net COPYONLY)
#configure_file(resources/startnet_seed_2.net startnet_seed_2.net COPYONLY)
#configure_file(resources/startnet_seed_3.net startnet_seed_3.net COPYONLY)
#configure_file(resources/startnet_seed_4.net startnet_seed_4.net COPYONLY)
#configure_file(resources/startnet_seed_5.net startnet_seed_5.net COPYONLY)
#configure_file(resources/startnet_seed_6.net startnet_seed_6.net COPYONLY)
#configure_file(resources/startnet_seed_7.net startnet_seed_7.net COPYONLY)
#configure_file(resources/startnet_seed_8.net startnet_seed_8.net COPYONLY)
#configure_file(resources/startnet_seed_9.net startnet_seed_9.net COPYONLY)
#configure_file(resources/startnet_seed_10.net startnet_seed_10.net COPYONLY)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(code main.cpp
        npp/npp.h
        npp/npp.cpp)
