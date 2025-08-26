add_executable(vit vit.cpp)
target_link_libraries(vit PUBLIC common vaststreamx::core
                                   glog::glog    ${OpenCV_LIBS} stdc++fs)
install(TARGETS vit DESTINATION bin)

add_executable(vit_prof vit_prof.cpp)
target_link_libraries(vit_prof PUBLIC common glog::glog  vaststreamx::core ${OpenCV_LIBS})
install(TARGETS vit_prof DESTINATION bin)
