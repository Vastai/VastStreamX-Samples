add_executable(image image.cpp)
target_link_libraries(image PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS image DESTINATION bin)

add_executable(tensor tensor.cpp)
target_link_libraries(tensor PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS tensor DESTINATION bin)