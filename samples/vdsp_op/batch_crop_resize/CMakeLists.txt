add_executable(batch_crop_resize batch_crop_resize.cpp)
target_link_libraries(batch_crop_resize PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS batch_crop_resize DESTINATION bin)

