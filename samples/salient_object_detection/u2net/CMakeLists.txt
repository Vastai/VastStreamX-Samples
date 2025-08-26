add_executable(u2net u2net.cpp)
target_link_libraries(u2net PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS u2net DESTINATION bin)

add_executable(u2net_prof u2net_prof.cpp)
target_link_libraries(u2net_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS u2net_prof DESTINATION bin)
