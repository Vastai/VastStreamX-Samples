include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tracker_cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tracker_cpp/ByteTracker/include)
include_directories(/usr/include/eigen3)
file(GLOB BYTETRACK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tracker_cpp/ByteTracker/src/*.cpp)

add_executable(decode_mot_encode_multi decode_mot_encode_multi.cpp ${BYTETRACK_SRC})
target_link_libraries(decode_mot_encode_multi PUBLIC common vaststreamx::vaststreamx ${OpenCV_LIBS} glog::glog)
install(TARGETS decode_mot_encode_multi DESTINATION bin)