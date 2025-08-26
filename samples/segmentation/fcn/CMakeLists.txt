add_executable(fcn fcn.cpp)
target_link_libraries(fcn PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS fcn DESTINATION bin)

add_executable(fcn_prof fcn_prof.cpp)
target_link_libraries(fcn_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS fcn_prof DESTINATION bin)

