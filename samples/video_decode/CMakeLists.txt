add_executable(video_decode video_decode.cpp)
target_link_libraries(video_decode PUBLIC common vaststreamx::core vaststreamx::media
                                   glog::glog   stdc++fs)
install(TARGETS video_decode DESTINATION bin)

add_executable(video_decode_prof video_decode_prof.cpp)
target_link_libraries(video_decode_prof PUBLIC common glog::glog  vaststreamx::core 
                       vaststreamx::media  stdc++fs)
install(TARGETS video_decode_prof DESTINATION bin)