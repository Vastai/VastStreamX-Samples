add_executable(video_writer video_writer.cpp)
target_link_libraries(video_writer PUBLIC common vaststreamx::core vaststreamx::media
                                   glog::glog    stdc++fs)
install(TARGETS video_writer DESTINATION bin)
