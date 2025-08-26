add_executable(dbnet dbnet.cpp dbnet_detector/clipper.cpp)
target_link_libraries(dbnet PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog)
install(TARGETS dbnet DESTINATION bin)

add_executable(dbnet_prof dbnet_prof.cpp dbnet_detector/clipper.cpp)
target_link_libraries(dbnet_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS dbnet_prof DESTINATION bin)
