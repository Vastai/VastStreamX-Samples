add_executable(flip flip.cpp)
target_link_libraries(flip PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS flip DESTINATION bin)