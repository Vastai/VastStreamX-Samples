add_executable(face_detection face_detection.cpp)
target_link_libraries(face_detection PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS face_detection DESTINATION bin)

add_executable(face_detection_prof face_detection_prof.cpp)
target_link_libraries(face_detection_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS face_detection_prof DESTINATION bin)
