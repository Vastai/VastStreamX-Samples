add_executable(ppocr ppocr.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_include_directories(ppocr PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(ppocr PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(ppocr PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS ppocr DESTINATION bin)

add_executable(ppocr_async ppocr_async.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_include_directories(ppocr_async PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(ppocr_async PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(ppocr_async PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS ppocr_async DESTINATION bin)

add_executable(doc_img_orient_cls doc_img_orient_cls.cpp)
target_include_directories(doc_img_orient_cls PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(doc_img_orient_cls PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(doc_img_orient_cls PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS doc_img_orient_cls DESTINATION bin)


add_executable(doc_img_orient_cls_prof doc_img_orient_cls_prof.cpp)
target_include_directories(doc_img_orient_cls_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(doc_img_orient_cls_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(doc_img_orient_cls_prof PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS doc_img_orient_cls_prof DESTINATION bin)

add_executable(text_det text_det.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_include_directories(text_det PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(text_det PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(text_det PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_det DESTINATION bin)

add_executable(text_det_prof text_det_prof.cpp ${CMAKE_SOURCE_DIR}/common/text_det_post/clipper.cpp)
target_include_directories(text_det_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(text_det_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(text_det_prof PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_det_prof DESTINATION bin)

add_executable(text_cls text_cls.cpp)
target_include_directories(text_cls PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(text_cls PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(text_cls PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_cls DESTINATION bin)

add_executable(text_cls_prof text_cls_prof.cpp)
target_include_directories(text_cls_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(text_cls_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(text_cls_prof PUBLIC common vsx-core glog::glog ${OpenCV_LIBS})
install(TARGETS text_cls_prof DESTINATION bin)

add_executable(text_rec text_rec.cpp)
target_include_directories(text_rec PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(text_rec PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(text_rec PUBLIC common vsx-core glog::glog ${OpenCV_LIBS} stdc++fs)
install(TARGETS text_rec DESTINATION bin)

add_executable(text_rec_prof text_rec_prof.cpp)
target_include_directories(text_rec_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(text_rec_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(text_rec_prof PUBLIC common vsx-core glog::glog ${OpenCV_LIBS}  stdc++fs)
install(TARGETS text_rec_prof DESTINATION bin)









