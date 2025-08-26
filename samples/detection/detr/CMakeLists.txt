add_executable(detr detr.cpp)
target_link_libraries(detr PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS detr DESTINATION bin)

add_executable(detr_prof detr_prof.cpp)
target_link_libraries(detr_prof PUBLIC common vaststreamx::core  glog::glog)
install(TARGETS detr_prof DESTINATION bin)

