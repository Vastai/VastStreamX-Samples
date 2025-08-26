add_executable(crop crop.cpp)
target_link_libraries(crop PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS crop DESTINATION bin)

