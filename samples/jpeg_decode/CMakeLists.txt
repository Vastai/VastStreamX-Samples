add_executable(jpeg_decode jpeg_decode.cpp)
target_link_libraries(jpeg_decode PUBLIC common vaststreamx::media stdc++fs ${OpenCV_LIBS} glog::glog)
install(TARGETS jpeg_decode DESTINATION bin)

add_executable(jpeg_decode_prof jpeg_decode_prof.cpp)
target_link_libraries(jpeg_decode_prof PUBLIC common glog::glog  vaststreamx::core 
                       vaststreamx::media  stdc++fs)
install(TARGETS jpeg_decode_prof DESTINATION bin)
