add_executable(buildin_op_prof buildin_op_prof.cpp)
target_link_libraries(buildin_op_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS buildin_op_prof DESTINATION bin)