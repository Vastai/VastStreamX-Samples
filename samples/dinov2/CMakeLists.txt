find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

add_executable(dinov2 dinov2.cpp)
target_include_directories(dinov2 PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(dinov2 PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(dinov2 PUBLIC vsx-core common ${OpenCV_LIBS} stdc++fs glog::glog Python3::Python pybind11::module)
install(TARGETS dinov2 DESTINATION bin)


add_executable(dinov2_prof dinov2_prof.cpp)
target_include_directories(dinov2_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/include)
target_link_directories(dinov2_prof PUBLIC /opt/vastai/vaststreamx/vaststreamx/lib)
target_link_libraries(dinov2_prof PUBLIC vsx-core common ${OpenCV_LIBS} glog::glog)
install(TARGETS dinov2_prof DESTINATION bin)
