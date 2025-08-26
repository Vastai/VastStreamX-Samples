cmake_minimum_required(VERSION 3.2)
project(vaststreamx-samples)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wno-class-memaccess)
if(RELEASE_BUILD)
  # safe compile
	set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG=1 -D_NDEBUG=1 -D_FORTIFY_SOURCE=2 -O2 ${CMAKE_C_FLAGS_RELEASE}")
	set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG=1 -D_NDEBUG=1 -D_FORTIFY_SOURCE=2 -O2 ${CMAKE_CXX_FLAGS_RELEASE}")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fpie -fstack-protector-all")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpie -fstack-protector-all")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pie -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
	# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pie -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
  # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXECUTABLE_LINKER_FLAGS} -s")
else()
	if (ADDRESS_SANITIZER STREQUAL "1")
		set(SANITIZE_FLAGS "-fsanitize=address") # -fsanitize=leak -fsanitize=undefined")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZE_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
  endif()
  set(CMAKE_C_FLAGS_DEBUG "-DDEBUG=1 -D_DEBUG=1 ${CMAKE_C_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG=1 -D_DEBUG=1 ${CMAKE_C_FLAGS_DEBUG}")
  add_definitions("-Wall -g")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/vaststreamx-samples)

find_package(vaststreamx REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

add_subdirectory(common)
add_subdirectory(samples)
