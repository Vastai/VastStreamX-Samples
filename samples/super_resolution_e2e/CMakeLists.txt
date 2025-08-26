add_executable(super_resolution_e2e super_resolution_e2e.cpp)
target_link_libraries(super_resolution_e2e PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS super_resolution_e2e DESTINATION bin)

add_executable(sr_e2e_prof sr_e2e_prof.cpp)
target_link_libraries(sr_e2e_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS sr_e2e_prof DESTINATION bin)