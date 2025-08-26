add_executable(argmax argmax.cpp)
target_link_libraries(argmax PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS argmax DESTINATION bin)

add_executable(argmax_prof argmax_prof.cpp)
target_link_libraries(argmax_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS argmax_prof DESTINATION bin)



