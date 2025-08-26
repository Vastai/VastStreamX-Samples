set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(center_point center_point.cpp)
target_link_libraries(center_point PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS center_point DESTINATION bin)

add_executable(center_point_prof center_point_prof.cpp)
target_link_libraries(center_point_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS center_point_prof DESTINATION bin)