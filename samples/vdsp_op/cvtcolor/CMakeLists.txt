add_executable(cvtcolor cvtcolor.cpp)
target_link_libraries(cvtcolor PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS cvtcolor DESTINATION bin)
