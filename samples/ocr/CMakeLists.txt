add_executable(crnn crnn.cpp)
target_link_libraries(crnn PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs   glog::glog)
install(TARGETS crnn DESTINATION bin)

add_executable(crnn_prof crnn_prof.cpp)
target_link_libraries(crnn_prof PUBLIC common vaststreamx::core  ${OpenCV_LIBS} glog::glog)
install(TARGETS crnn_prof DESTINATION bin)