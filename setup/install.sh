#!/bin/bash

SRC_DIR=$HOME/packages
INSTALL_DIR=$HOME/local

rm -rf $INSTALL_DIR $SRC_DIR
mkdir -p $INSTALL_DIR $SRC_DIR

# Install CMake
cd $SRC_DIR;
git clone https://gitlab.kitware.com/cmake/cmake.git;
cd cmake;
git fetch --all --tags --prune;
git checkout tags/v3.28.1;
./bootstrap --prefix=$INSTALL_DIR/cmake -- -DCMAKE_USE_OPENSSL=OFF;
make -j;
make install -j;

# Install OpenBLAS
cd $SRC_DIR;
git clone https://github.com/OpenMathLib/OpenBLAS.git;
cd OpenBLAS;
git fetch --all --tags --prune;
git checkout tags/v0.3.26;
make -j;
make install -j PREFIX=$INSTALL_DIR/openblas;

# Install GNU libtool
cd $SRC_DIR;
wget https://ftpmirror.gnu.org/libtool/libtool-2.4.7.tar.gz
tar xf libtool-2.4.7.tar.gz;
mkdir -p libtool-2.4.7/build;
cd libtool-2.4.7/build;
../configure --prefix=$INSTALL_DIR/libtool;
make -j;
make install -j;

export PATH=$HOME/local/libtool/bin:$PATH

# Install FxT
cd $SRC_DIR;
wget http://download.savannah.nongnu.org/releases/fkt/fxt-0.3.14.tar.gz
tar xf fxt-0.3.14.tar.gz;
mkdir -p fxt-0.3.14/build;
cd fxt-0.3.14/build;
../configure --prefix=$INSTALL_DIR/fxt;
make -j;
make install -j;

# Install Boost for SimGrid
#cd $SRC_DIR;
#wget https://boostorg.jfrog.io/artifactory/main/release/1.84.0/source/boost_1_84_0.tar.gz
#tar xf boost_1_84_0.tar.gz;
#cd boost_1_84_0/
#./bootstrap.sh
#./b2 install --prefix=$INSTALL_DIR/boost

# Install SimGrid
#cd $SRC_DIR;
#wget https://framagit.org/simgrid/simgrid/-/archive/v3.30/simgrid-v3.30.tar.gz
#tar xf simgrid-v3.30.tar.gz;
#mkdir -p simgrid-v3.30/build;
#cd simgrid-v3.30/build;
#cmake --DBOOST_ROOT=$INSTALL_DIR/boost  --DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/simgrid .;
#make -j;
#make install -j;

#export PKG_CONFIG_PATH=$PKG_CONFIG_FILE:~/local/simgrid/lib/ppkgconfig/simgrid.pc


# Install New Madeleine
#cd $SRC_DIR
#git clone https://gitlab.inria.fr/pm2/pm2.git
#cd pm2
#git checkout 7e3edd5002885d4a114c5e551106f28ae9630e01
#cd scripts
#./pm2-build-packages ./madmpi.conf --prefix=$INSTALL_DIR/nmad

# Install StarPU
cd $SRC_DIR;
git clone --recurse-submodules https://gitlab.inria.fr/starpu/starpu.git;
cd starpu;
git fetch --all --tags --prune;
git checkout starpu-1.4;
./autogen.sh;
mkdir build;
cd build;
../configure --prefix=$INSTALL_DIR/starpu --disable-opencl --disable-build-doc --disable-build-examples --disable-build-test;
make -j;
make install -j;
make clean;
../configure --prefix=$INSTALL_DIR/starpu-fxt --disable-opencl --disable-build-doc --disable-build-examples --disable-build-test --with-fxt=$INSTALL_DIR/fxt;
make -j;
make install -j;
make clean;
../configure --prefix=$INSTALL_DIR/starpu-parallel --disable-opencl --disable-build-doc --disable-build-examples --disable-build-test --enable-parallel-worker;
make -j;
make install -j;
make clean;


