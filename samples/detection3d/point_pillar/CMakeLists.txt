add_executable(point_pillar point_pillar.cpp)
target_link_libraries(point_pillar PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS point_pillar DESTINATION bin)

add_executable(point_pillar_prof point_pillar_prof.cpp)
target_link_libraries(point_pillar_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS point_pillar_prof DESTINATION bin)