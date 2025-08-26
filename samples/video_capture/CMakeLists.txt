add_executable(video_capture video_capture.cpp)
target_link_libraries(video_capture PUBLIC common vaststreamx::media ${OpenCV_LIBS} glog::glog)
install(TARGETS video_capture DESTINATION bin)

