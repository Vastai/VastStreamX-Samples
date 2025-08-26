add_executable(norma_tensor_3ch norma_tensor_3ch.cpp)
target_link_libraries(norma_tensor_3ch PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS norma_tensor_3ch DESTINATION bin)

add_executable(norma_tensor_3ch_prof norma_tensor_3ch_prof.cpp)
target_link_libraries(norma_tensor_3ch_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS norma_tensor_3ch_prof DESTINATION bin)

