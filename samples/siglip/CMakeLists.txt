add_executable(siglip_sample siglip_sample.cpp)
target_link_libraries(siglip_sample PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog stdc++fs)
install(TARGETS siglip_sample DESTINATION bin)

add_executable(siglip_image_prof siglip_image_prof.cpp)
target_link_libraries(siglip_image_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS siglip_image_prof DESTINATION bin)

add_executable(siglip_text_prof siglip_text_prof.cpp)
target_link_libraries(siglip_text_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS siglip_text_prof DESTINATION bin)

