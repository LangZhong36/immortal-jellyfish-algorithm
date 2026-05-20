cmake_minimum_required(VERSION 3.14)
project(ImmortalJellyfishAlgorithm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -Wall -Wextra")

include_directories(include)

add_library(ija_core STATIC src/ija.cpp)

add_executable(ija_main  src/main.cpp)
target_link_libraries(ija_main  PRIVATE ija_core)

add_executable(ija_demo  src/benchmark_demo.cpp)
target_link_libraries(ija_demo  PRIVATE ija_core)

install(TARGETS ija_main ija_demo RUNTIME DESTINATION bin)
install(FILES include/ija.h DESTINATION include)

enable_testing()
add_test(NAME SmokeTest COMMAND ija_demo)
