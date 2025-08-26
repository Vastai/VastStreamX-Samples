add_executable(brightness brightness.cpp)
target_link_libraries(brightness PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS brightness DESTINATION bin)

add_executable(brightness_prof brightness_prof.cpp)
target_link_libraries(brightness_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS brightness_prof DESTINATION bin)
