add_executable(detection detection.cpp)
target_link_libraries(detection PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS detection DESTINATION bin)


add_executable(det_prof det_prof.cpp)
target_link_libraries(det_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS det_prof DESTINATION bin)
