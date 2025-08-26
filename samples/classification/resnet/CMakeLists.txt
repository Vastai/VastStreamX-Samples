add_executable(classification classification.cpp)
target_link_libraries(classification PUBLIC common vaststreamx::core
                                   glog::glog    ${OpenCV_LIBS} stdc++fs)
install(TARGETS classification DESTINATION bin)

add_executable(cls_prof cls_prof.cpp)
target_link_libraries(cls_prof PUBLIC common glog::glog  vaststreamx::core ${OpenCV_LIBS})
install(TARGETS cls_prof DESTINATION bin)