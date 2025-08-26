add_executable(decode_detection_encode decode_detection_encode.cpp)
target_link_libraries(decode_detection_encode PUBLIC common vaststreamx::vaststreamx ${OpenCV_LIBS} glog::glog)
install(TARGETS decode_detection_encode DESTINATION bin)


