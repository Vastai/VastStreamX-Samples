add_executable(scale scale.cpp)
target_link_libraries(scale PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS scale DESTINATION bin)

