add_executable(bisenet bisenet.cpp)
target_link_libraries(bisenet PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS bisenet DESTINATION bin)

add_executable(bisenet_prof bisenet_prof.cpp)
target_link_libraries(bisenet_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS bisenet_prof DESTINATION bin)
