add_executable(decode_and_detection decode_and_detection.cpp)
target_link_libraries(decode_and_detection PUBLIC common vaststreamx::vaststreamx ${OpenCV_LIBS} glog::glog)
install(TARGETS decode_and_detection DESTINATION bin)


