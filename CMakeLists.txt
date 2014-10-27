cmake_minimum_required (VERSION 2.8)

project (rgbd)
find_package( OpenCV REQUIRED )

#fix the odd linking issue
#-fPIC needed for shared library

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
add_definitions("-g")
add_definitions("-Wall")
add_definitions("-std=c++11")
add_definitions("-O3")


include_directories(src)
add_executable(rgdb src/main.cpp)
target_link_libraries(rgdb
  ${OpenCV_LIBS} )

