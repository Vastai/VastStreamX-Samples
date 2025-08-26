include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tracker_cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tracker_cpp/ByteTracker/include)
include_directories(/usr/include/eigen3)
file(GLOB BYTETRACK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tracker_cpp/ByteTracker/src/*.cpp)

add_executable(bytetracker bytetracker.cpp ${BYTETRACK_SRC})
target_link_libraries(bytetracker PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog  )
install(TARGETS bytetracker DESTINATION bin)


add_executable(bytetracker_prof bytetracker_prof.cpp ${BYTETRACK_SRC})
target_link_libraries(bytetracker_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS}  glog::glog )
install(TARGETS bytetracker_prof DESTINATION bin)
