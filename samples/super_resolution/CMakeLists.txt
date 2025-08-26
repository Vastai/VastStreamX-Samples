add_executable(super_resolution super_resolution.cpp)
target_link_libraries(super_resolution PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS super_resolution DESTINATION bin)


add_executable(sr_prof sr_prof.cpp)
target_link_libraries(sr_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS sr_prof DESTINATION bin)

