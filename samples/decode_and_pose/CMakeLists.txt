find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(decode_and_pose decode_and_pose.cpp libtorch_postprocess.cpp cnpy.cpp)

target_link_libraries(
    decode_and_pose 
    PUBLIC 
    common 
    vaststreamx::vaststreamx 
    TorchVision::TorchVision 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS} 
    glog::glog
    stdc++fs)

install(TARGETS decode_and_pose DESTINATION bin)


