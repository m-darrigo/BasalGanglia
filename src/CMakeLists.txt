cmake_minimum_required (VERSION 3.13.4)
project (basal_ganglia_model)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-pedantic -Wall -Wextra -O3")

add_library(model STATIC model.cpp)
target_link_libraries(model yaml-cpp)

add_executable(main main.cpp)
target_link_libraries(main model)
target_link_libraries(main yaml-cpp)