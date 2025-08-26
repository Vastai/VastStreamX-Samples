add_executable(rtdetr rtdetr.cpp)
target_link_libraries(rtdetr PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS rtdetr DESTINATION bin)

add_executable(rtdetr_prof rtdetr_prof.cpp)
target_link_libraries(rtdetr_prof PUBLIC common vaststreamx::core  glog::glog)
install(TARGETS rtdetr_prof DESTINATION bin)