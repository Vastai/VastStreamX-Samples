add_executable(card_info card_info.cpp)
target_link_libraries(card_info PUBLIC  vaststreamx::core  glog::glog)
install(TARGETS card_info DESTINATION bin)