add_executable(alpr alpr.cpp)
target_link_libraries(alpr PUBLIC common vaststreamx::core ${OpenCV_LIBS}
                                  stdc++fs glog::glog)
install(TARGETS alpr DESTINATION bin)
