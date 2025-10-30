add_executable(decode_detection_encode_multi decode_detection_encode_multi.cpp)
target_link_libraries(decode_detection_encode_multi PUBLIC common vaststreamx::vaststreamx ${OpenCV_LIBS} glog::glog)
install(TARGETS decode_detection_encode_multi DESTINATION bin)


add_executable(decode_detection_writer_multi decode_detection_writer_multi.cpp)
target_link_libraries(decode_detection_writer_multi PUBLIC common vaststreamx::vaststreamx ${OpenCV_LIBS} glog::glog)
install(TARGETS decode_detection_writer_multi DESTINATION bin)

