cmake_minimum_required(VERSION 2.9)
project(test_faster_rcnn)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES test_faster_rcnn.cpp Caffe_Net.cpp Caffe_Net.h config.h)
add_executable(test_faster_rcnn ${SOURCE_FILES})

target_link_libraries(test_faster_rcnn ${OpenCV_LIBS} )
target_link_libraries(test_faster_rcnn ${Caffe_LIBRARIES})
