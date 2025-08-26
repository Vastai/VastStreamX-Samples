add_executable(clip_sample clip_sample.cpp)
target_link_libraries(clip_sample PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog stdc++fs)
install(TARGETS clip_sample DESTINATION bin)

add_executable(clip_image_prof clip_image_prof.cpp)
target_link_libraries(clip_image_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS clip_image_prof DESTINATION bin)

add_executable(clip_text_prof clip_text_prof.cpp)
target_link_libraries(clip_text_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS clip_text_prof DESTINATION bin)

