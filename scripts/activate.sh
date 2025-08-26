#! /bin/bash


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/vastai/vaststreamx/vaststreamx/lib
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/vastai/vaststreamx/vaststreamx:/opt/3rdparty
export GLOG_minloglevel=2
export GLOG_logtostderr=1
export VSX_DISABLE_DEEPBIND=1 
export VACM_LOG_CFG=/home

