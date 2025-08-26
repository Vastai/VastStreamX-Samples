add_executable(run_stream_async run_stream_async.cpp)
target_link_libraries(run_stream_async PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS run_stream_async DESTINATION bin)

add_executable(run_stream_async_prof run_stream_async_prof.cpp)
target_link_libraries(run_stream_async_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS run_stream_async_prof DESTINATION bin)
