add_executable(yolov8_seg yolov8_seg.cpp)
target_link_libraries(yolov8_seg PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS yolov8_seg DESTINATION bin)

add_executable(yolov8_seg_prof yolov8_seg_prof.cpp)
target_link_libraries(yolov8_seg_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS yolov8_seg_prof DESTINATION bin)


