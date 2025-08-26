add_executable(dynamic_yolo dynamic_yolo.cpp)
target_link_libraries(dynamic_yolo PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS dynamic_yolo DESTINATION bin)


add_executable(dynamic_yolo_prof dynamic_yolo_prof.cpp)
target_link_libraries(dynamic_yolo_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS dynamic_yolo_prof DESTINATION bin)


