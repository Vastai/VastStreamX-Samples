find_package(OpenMP REQUIRED) 
find_package(Torch REQUIRED)

message(STATUS "TORCH_FOUND:" ${TORCH_FOUND})
message(STATUS "TORCH_INCLUDE_DIRS:" ${TORCH_INCLUDE_DIRS})
message(STATUS "TORCH_LIBRARIES:" ${TORCH_LIBRARIES})
message(STATUS "TORCH_CXX_FLAGS:" ${TORCH_CXX_FLAGS})

add_executable(yolo_world yolo_world.cpp)
target_link_libraries(yolo_world PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog 
                        stdc++fs ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
install(TARGETS yolo_world DESTINATION bin)

add_executable(yolo_world_image_prof yolo_world_image_prof.cpp)
target_link_libraries(yolo_world_image_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} 
    glog::glog ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
install(TARGETS yolo_world_image_prof DESTINATION bin)

add_executable(yolo_world_text_prof yolo_world_text_prof.cpp)
target_link_libraries(yolo_world_text_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} 
    glog::glog ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
install(TARGETS yolo_world_text_prof DESTINATION bin)