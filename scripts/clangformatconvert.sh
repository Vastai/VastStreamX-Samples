#!/bin/bash

filelist=`git ls-files`

# function to check if C++ file (based on suffix)
# can probably be done much shorter
function checkCPP(){
    if [[ $1 == *.cc ]];then
		return 0
    elif [[ $1 == *.cpp ]];then
		return 0
    elif [[ $1 == *.cxx ]];then
		return 0
    elif [[ $1 == *.cu ]];then
		return 0
    elif [[ $1 == *.C ]];then
		return 0
    elif [[ $1 == *.c++ ]];then
		return 0
    elif [[ $1 == *.c ]];then
		return 0
    elif [[ $1 == *.CPP ]];then
		return 0
	# header files
    elif [[ $1 == *.h ]];then
		return 0
    elif [[ $1 == *.hpp ]];then
		return 0
    elif [[ $1 == *.hh ]];then
		return 0
    elif [[ $1 == *.cuh ]];then
		return 0
    elif [[ $1 == *.icc ]];then
		return 0
    fi
    return 1
}

# check list of files
for f in $filelist; do
    if checkCPP $f; then
				echo "CONVERTING MATCHING FILE ${f}"
				# apply the clang-format script
				clang-format -style=Google -i ${f}
    fi
done


exit 0
