add_executable(bert bert.cpp)
target_link_libraries(bert PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS bert DESTINATION bin)

add_executable(bert_prof bert_prof.cpp)
target_link_libraries(bert_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS bert_prof DESTINATION bin)