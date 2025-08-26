add_executable(mobilevit mobilevit.cpp)
target_link_libraries(mobilevit PUBLIC common vaststreamx::core glog::glog    ${OpenCV_LIBS} stdc++fs)
install(TARGETS mobilevit DESTINATION bin)

add_executable(mobilevit_prof mobilevit_prof.cpp)
target_link_libraries(mobilevit_prof PUBLIC common glog::glog  vaststreamx::core ${OpenCV_LIBS})
install(TARGETS mobilevit_prof DESTINATION bin)