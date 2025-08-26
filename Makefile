# format refrence github repo url
# https://github.com/muttleyxd/clang-tools-static-binaries
# https://github.com/cheshirekow/cmake_format/releases/tag/v0.6.13
# https://github.com/google/styleguide/blob/gh-pages/cpplint/cpplint.py
# https://github.com/psf/black use: pip install git+https://github.com/psf/black

CHECK_PATH=.

staged_cxx_files ?= $(shell git diff --staged --name-only --diff-filter=d -- '*.c' '*.h' '*.cpp' '*.cc' '*.hpp')
staged_cmake_files ?= $(shell git diff --staged --name-only --diff-filter=d -- '*CMakeLists.txt' '*.cmake' '*.cmake.in')
staged_py_files ?= $(shell git diff --staged --name-only --diff-filter=d --  '*.py')
staged_go_files ?= $(shell git diff --staged --name-only --diff-filter=d --  '*.go')

CPPLINT_FLAG := -build/c++11,-runtime/references,-readability/check,-runtime/string,-runtime/threadsafe_fn,-build/include,-whitespace/ending_newline,-whitespace/indent_namespace

STAGED_CPP_IGNORE_PATH := evaluation/% \
						  common/text_det_post/% \
						  samples/text_detection/dbnet_detector/% \
						  samples/multi_object_tracking/tracker_cpp/ByteTracker/%

staged_cxx_files := $(filter-out  $(STAGED_CPP_IGNORE_PATH), $(staged_cxx_files))
staged_py_files := $(filter-out evaluation/%, $(staged_py_files))

## ------enable cpplint for work-staged code
cpplint-staged: ${staged_cxx_files}
	@if [ ! $$(which cpplint) ]; then \
		pip3 install cpplint; \
	fi
	@if [ ! -z "$(staged_cxx_files)" ]; then \
		echo "cpplint c/cpp files ..."; \
		python3 .toolbox/add_copyright.py ${staged_cxx_files}; \
	    cpplint --filter=${CPPLINT_FLAG} ${staged_cxx_files}; \
	else \
		echo "no staged general c/cpp files to format"; \
	fi

## ------enable pylint for work-staged code
pylint-staged: $(staged_py_files)
	@if [ ! "$$(pylint --version)" ]; then \
		pip3 install pylint; \
	else \
		echo "pylint has already been installed!"; \
	fi
	@if [ ! -z "$(staged_py_files)" ]; then \
		echo "pylint py files ..."; \
		python3 .toolbox/add_copyright.py ${staged_py_files}; \
	    pylint --disable=all --enable=C -E --msg-template='{path}:{line}:{column}:error:[**[{msg_id}](http://pylint-messages.wikidot.com/messages:{msg_id})**] ({category}, {symbol})<br><br>{msg}' --output-format=parseable ${staged_py_files}; \
	else \
		echo "no staged general py files to format"; \
	fi

#############################################################################
# tidy
#############################################################################
IGNORE_PATH := -path ./.toolbox -prune -o \
 			   -path ./evaluation -prune -o \
			   -path ./build -prune -o \
			   -path ./samples/multi_object_tracking/tracker_cpp/ByteTracker -prune -o \
			   -path ./samples/elic/ELICUtilis -prune -o \
			   -path ./tests -prune -o \
			   -path ./samples/mlic++/modules -o 

# IGNORE_FILE := ! -iname Makefile -iname json.hpp 
IGNORE_FILE := ! -iname Makefile \
			   ! -path "./common/atomicops.h" \
			   ! -path "./common/cmdline.hpp" \
			   ! -path "./common/file_system.hpp" \
			   ! -path "./common/json.hpp" \
			   ! -path "./common/readerwritercircularbuffer.h" \
			   ! -path "./common/readerwriterqueue.h" \
			   ! -iname "clipper.h" \
			   ! -iname "clipper.cpp" \
			   ! -iname "db_post_process.hpp" \
    		   ! -path "common/npy-halffloat.h" \
			   ! -iname "text_det_post.hpp"

## ------format go cpp cmake and python for work-staged code
format-staged: format-staged-go format-staged-cpp format-staged-cmake format-staged-py

## ------format go for work-staged code
format-staged-go: $(staged_go_files)
	@if [ ! -z "$(staged_go_files)" ]; then \
		echo "formating general go files ..."; \
		echo $(staged_go_files) | xargs gofmt -w \
	else \
		echo "no staged general go files to format"; \
	fi

## ------format cpp for work-staged code
format-staged-cpp: $(staged_cxx_files)
	@if [ ! -z "$(staged_cxx_files)" ]; then \
		echo "formating general c/cpp files ..."; \
		.toolbox/clang-format-9_linux-amd64 --style=Google -i $(staged_cxx_files); \
	else \
		echo "no staged general c/cpp files to format"; \
	fi

## ------format cmake for work-staged code
format-staged-cmake: $(staged_cmake_files)
	@if [ ! -z "$(staged_cmake_files)" ]; then \
		echo "formating cmake files ..."; \
		.toolbox/cmake-format -i -c .cmake-format.py -- $(staged_cmake_files); \
	else \
		echo "no staged cmake files to format"; \
	fi

## ------format python for work-staged code
format-staged-py: $(staged_py_files)
	@if [[ ! $(pip3 list|grep black) ]];then pip3 install black;fi
	@if [ ! -z "$(staged_py_files)" ]; then \
		echo "formating py files ..."; \
		python3 -m yapf --style=pep8 -i $(staged_py_files); \
	else \
		echo "no staged py files to format"; \
	fi

## ------format cpp cmake python and go
format: format-cpp format-cmake format-py format-go

## ------format cpp eg: make format-cpp CHECK_PATH=./
format-cpp:
	find ${CHECK_PATH} ${IGNORE_PATH} ${IGNORE_FILE} \
				 -type f \( -iname '*.h'  -o \
				 -iname '*.cpp'  -o \
				 -iname '*.cc'  -o \
				 -iname '*.hpp'  -o \
				 -iname '*.c' \) -print \
				 -exec .toolbox/clang-format-9_linux-amd64 -i -style=Google {} \; \
    			 -exec python3 .toolbox/add_copyright.py {} \;


## ------format cmake eg: make format-cmake CHECK_PATH=./
format-cmake:
	find ${CHECK_PATH} ${IGNORE_PATH} ${IGNORE_FILE} \
				 -type f \( -iname CMakeLists.txt -o \
				 -iname "*.cmake" \) -print |xargs .toolbox/cmake-format -i -c .toolbox/.cmake-format.py --

## ------format go eg: make format-go CHECK_PATH=./
format-go:
	find ${CHECK_PATH} ${IGNORE_PATH} ${IGNORE_FILE} -type f \( -iname *.go \) |xargs gofmt -w

## ------format python eg: make format-py CHECK_PATH=./
format-py:
	@if [ ! $$(black --help) ]; then \
		pip3 install black; \
	else \
		echo "Black has already been installed!"; \
	fi
	find ${CHECK_PATH} ${IGNORE_PATH} ${IGNORE_FILE} -type f \( -iname '*.py' \) \
		|xargs -i bash -c "if [ -f {} ];then echo {};fi" \
		|xargs black

## ------enable cpplint eg: make cpplint CHECK_PATH=./
cpplint:
	find ${CHECK_PATH} ${IGNORE_PATH} ${IGNORE_FILE} \
				 -type f \( -iname '*.h'  -o \
				 -iname '*.cpp'  -o \
				 -iname '*.cc'  -o \
				 -iname '*.hpp'  -o \
				 -iname '*.c' \) -print \
				 | xargs cpplint --filter=${CPPLINT_FLAG}

## ------enable cpplint eg: make cpplint CHECK_PATH=./ 
pylint:
	@if [ ! $$(pylint --version) ]; then \
		pip3 install pylint; \
	else \
		echo "pylint has already been installed!"; \
	fi
	find ${CHECK_PATH} ${IGNORE_PATH} ${IGNORE_FILE} \
				 -type f \( -iname "*.py" \) | \
				 xargs -i pylint --disable=all --enable=C -E --msg-template='{path}:{line}:{column}:error:[**[{msg_id}](http://pylint-messages.wikidot.com/messages:{msg_id})**] ({category}, {symbol})<br><br>{msg}' --output-format=parseable {}

## ------add hook
add_hook:
	@.toolbox/.hooks/install_hooks.sh

## ------download toolbox and githook
download:
	@wget -O toolbox.tar.gz http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/pipelineartifact?artifactid=181007
	@tar -zxvf toolbox.tar.gz && rm -f toolbox.tar.gz

## ------download toolbox githook and add hook
install: download add_hook

################################################################################
# Help
################################################################################
TARGET_MAX_CHAR_NUM=50
## Show help
help:
	@echo ''
	@echo 'Usage:'
	@echo ' make Target'
	@echo ''
	@echo ${HOSTARCH}
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
    helpMessage = match(lastLine, /^## (.*)/); \
    if (helpMessage) { \
      helpCommand = substr($$1, 0, index($$1, ":")-1); \
      helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
      printf " %-$(TARGET_MAX_CHAR_NUM)s: %s\n", helpCommand, helpMessage; \
    } \
  } \
  { lastLine = $$0 }' $(MAKEFILE_LIST)
