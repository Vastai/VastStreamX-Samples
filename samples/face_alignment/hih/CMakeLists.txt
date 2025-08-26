add_executable(hih hih.cpp)
target_link_libraries(hih PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS hih DESTINATION bin)

add_executable(hih_prof hih_prof.cpp)
target_link_libraries(hih_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS hih_prof DESTINATION bin)
