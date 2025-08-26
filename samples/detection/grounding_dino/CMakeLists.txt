find_package(OpenBLAS REQUIRED)
add_executable(grounding_dino grounding_dino.cpp)
target_link_libraries(grounding_dino PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog OpenBLAS::OpenBLAS)
install(TARGETS grounding_dino DESTINATION bin)


add_executable(grounding_dino_text_enc_prof grounding_dino_text_enc_prof.cpp)
target_link_libraries(grounding_dino_text_enc_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog OpenBLAS::OpenBLAS)
install(TARGETS grounding_dino_text_enc_prof DESTINATION bin)

add_executable(grounding_dino_image_enc_prof grounding_dino_image_enc_prof.cpp)
target_link_libraries(grounding_dino_image_enc_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog OpenBLAS::OpenBLAS)
install(TARGETS grounding_dino_image_enc_prof DESTINATION bin)
