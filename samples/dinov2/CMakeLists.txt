find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# set(PYBIND11_PATH /opt/conda/envs/python38/lib/python3.8/site-packages/pybind11/)
set(PYBIND11_PATH /usr/local/lib/python3.8/dist-packages/pybind11)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
find_package(pybind11 CONFIG REQUIRED)  

add_executable(dinov2 dinov2.cpp)
target_link_libraries(dinov2 PUBLIC common vaststreamx::core ${OpenCV_LIBS} stdc++fs glog::glog Python3::Python pybind11::module)
install(TARGETS dinov2 DESTINATION bin)


add_executable(dinov2_prof dinov2_prof.cpp)
target_link_libraries(dinov2_prof PUBLIC common vaststreamx::core ${OpenCV_LIBS} glog::glog)
install(TARGETS dinov2_prof DESTINATION bin)


