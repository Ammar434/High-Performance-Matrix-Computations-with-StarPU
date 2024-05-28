#!/bin/bash

SRC_DIR=$HOME/packages
INSTALL_DIR=$HOME/local

# Install StarPU
cd $SRC_DIR;
#git clone --recurse-submodules https://gitlab.inria.fr/starpu/starpu.git;
cd starpu;
git fetch --all --tags --prune;
git checkout starpu-1.4;
./autogen.sh;
mkdir build;
cd build;
../configure --prefix=$INSTALL_DIR/starpu --disable-opencl --disable-build-doc --disable-build-examples --disable-build-test --enable-bubble --enable-openmp --enable-parallel-worker;
make -j;
make install -j;
make clean;

