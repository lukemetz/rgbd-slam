cmake_minimum_required (VERSION 2.8)

project (rgbd)
find_package(OpenCV REQUIRED)
find_package(Qt4 REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
add_definitions("-g")
#add_definitions("-Wall")
add_definitions("-std=c++11")
add_definitions("-O3")

include(${QT_USE_FILE})
include_directories(src)
include_directories(lib/qcustomplot)

aux_source_directory(src SOURCE)

add_executable(rgdb
  ${SOURCE}
)

target_link_libraries(rgdb
  ${OpenCV_LIBS}
  ${QT_LIBRARIES}
  ${CMAKE_SOURCE_DIR}/lib/qcustomplot/libqcustomplot.a
)


