#!/bin/sh

mkdir build

cd build

cmake .. -DUSE_MPI=ON -DMPI_CXX_INCLUDE_PATH=/tools/openmpi/include -DMPI_CXX_COMPILER=/tools/openmpi/bin/mppicxx -DMPI_LIBRARY="/tools/openmpi/lib/libmpi_cxx.so;/tools/openmpi/lib/libmpi.so" -DMPI_EXTRA_LIBRARY=/usr/local/lib/libpmi.so -DLMDB_INCLUDE_DIR=/mnt/newhome/lxc/tools/liblmdb -DLMDB_LIBRARIES=/mnt/newhome/lxc/tools/liblmdb/liblmdb.so -DLevelDB_INCLUDE=/mnt/newhome/lxc/tools/leveldb/include -DLevelDB_LIBRARY=/mnt/newhome/lxc/tools/leveldb/libleveldb.so -DBLAS=mkl -DBoost_DIR=/tools/boost -DCUDA_TOOLKIT_ROOT_DIR=/tools/cuda -DCUDA_SDK_ROOT_DIR=/tools/cuda/samples -DGFLAGS_ROOT_DIR=/mnt/newhome/lxc/tools/glfags -DGFLAGS_INCLUDE_DIR=/mnt/newhome/lxc/tools/gflags/include -DGFLAGS_LIBRARY=/mnt/newhome/lxc/tools/gflags/lib/libgflags.so -DGLOG_ROOT_DIR=/mnt/newhome/lxc/tools/glog -DGLOG_INCLUDE_DIR=/mnt/newhome/lxc/tools/glog/include -DGLOG_LIBRARY=/mnt/newhome/lxc/tools/glog/lib/libglog.so -DUSE_CUDNN=ON -DCUDNN_ROOT=/mnt/newhome/lxc/tools/cudnn -DMKL_ROOT=/tools/mkl -DOpenCV_DIR=/tools/opencv/install_dir/share/OpenCV -DOpenCV_LIB_DIR_OPT=/tools/opencv/install_dir/lib

make -j && make install

cd ..
