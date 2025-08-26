add_executable(warpaffine warpaffine.cpp)
target_link_libraries(warpaffine PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS warpaffine DESTINATION bin)

