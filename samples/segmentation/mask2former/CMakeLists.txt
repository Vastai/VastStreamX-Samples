add_executable(mask2former mask2former.cpp)
target_link_libraries(mask2former PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS mask2former DESTINATION bin)

add_executable(mask2former_prof mask2former_prof.cpp)
target_link_libraries(mask2former_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS mask2former_prof DESTINATION bin)