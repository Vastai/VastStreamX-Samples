add_executable(isnet isnet.cpp)
target_link_libraries(isnet PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS isnet DESTINATION bin)

add_executable(isnet_prof isnet_prof.cpp)
target_link_libraries(isnet_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS isnet_prof DESTINATION bin)
