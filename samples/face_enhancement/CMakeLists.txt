add_executable(face_enhancement face_enhancement.cpp)
target_link_libraries(face_enhancement PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS face_enhancement DESTINATION bin)


add_executable(face_enhancement_prof face_enhancement_prof.cpp)
target_link_libraries(face_enhancement_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS})
install(TARGETS face_enhancement_prof DESTINATION bin)

