add_executable(ocr_e2e ocr_e2e.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_link_libraries(ocr_e2e PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS ocr_e2e DESTINATION bin)

add_executable(text_det text_det.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_link_libraries(text_det PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_det DESTINATION bin)

add_executable(text_det_prof text_det_prof.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_link_libraries(text_det_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_det_prof DESTINATION bin)

add_executable(text_cls_prof text_cls_prof.cpp)
target_link_libraries(text_cls_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS})
install(TARGETS text_cls_prof DESTINATION bin)

add_executable(text_rec text_rec.cpp)
target_link_libraries(text_rec PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_rec DESTINATION bin)

add_executable(text_rec_prof text_rec_prof.cpp)
target_link_libraries(text_rec_prof PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS}  stdc++fs)
install(TARGETS text_rec_prof DESTINATION bin)

add_executable(ocr_e2e_async ocr_e2e_async.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_link_libraries(ocr_e2e_async PUBLIC common vaststreamx::core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS ocr_e2e_async DESTINATION bin)
