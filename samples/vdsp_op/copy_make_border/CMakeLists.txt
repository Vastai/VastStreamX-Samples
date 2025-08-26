add_executable(copy_make_border copy_make_border.cpp)
target_link_libraries(copy_make_border PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS copy_make_border DESTINATION bin)