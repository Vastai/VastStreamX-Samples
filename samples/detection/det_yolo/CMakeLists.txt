add_executable(detection detection.cpp)
target_include_directories(detection PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(detection PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(detection PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS detection DESTINATION bin)


add_executable(det_prof det_prof.cpp)
target_include_directories(det_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(det_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(det_prof PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS det_prof DESTINATION bin)
