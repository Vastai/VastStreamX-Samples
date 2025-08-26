add_executable(face_recognition face_recognition.cpp)
target_link_libraries(face_recognition PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS face_recognition DESTINATION bin)

add_executable(face_recognition_prof face_recognition_prof.cpp)
target_link_libraries(face_recognition_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS face_recognition_prof DESTINATION bin)

