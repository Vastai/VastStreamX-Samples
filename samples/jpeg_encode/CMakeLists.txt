add_executable(jpeg_encode jpeg_encode.cpp)
target_link_libraries(jpeg_encode PUBLIC common vaststreamx::media ${OpenCV_LIBS} glog::glog)
install(TARGETS jpeg_encode DESTINATION bin)

add_executable(jpeg_encode_prof jpeg_encode_prof.cpp)
target_link_libraries(jpeg_encode_prof PUBLIC common glog::glog  vaststreamx::core 
                       vaststreamx::media  stdc++fs)
install(TARGETS jpeg_encode_prof DESTINATION bin)