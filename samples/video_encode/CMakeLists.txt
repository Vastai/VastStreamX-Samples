add_executable(video_encode video_encode.cpp)
target_link_libraries(video_encode PUBLIC common vaststreamx::core vaststreamx::media
                                   glog::glog    stdc++fs)
install(TARGETS video_encode DESTINATION bin)

add_executable(video_encode_prof video_encode_prof.cpp)
target_link_libraries(video_encode_prof PUBLIC common glog::glog  vaststreamx::core 
                       vaststreamx::media  stdc++fs)
install(TARGETS video_encode_prof DESTINATION bin)