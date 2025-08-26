add_executable(image_process image_process.cpp)
target_link_libraries(image_process PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS image_process DESTINATION bin)

