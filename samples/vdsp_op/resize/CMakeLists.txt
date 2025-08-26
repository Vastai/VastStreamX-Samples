add_executable(resize resize.cpp)
target_link_libraries(resize PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS resize DESTINATION bin)

