#!/usr/bin/env bash 

set -e

if [ -z "$MKLROOT" ]; then
    echo "Set the MKLROOT environment variable to the root of your Intel MKL directory,"
    echo "for example:"
    echo "             Linux: export MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl"
    echo "             Mac:   export MKLROOT=/opt/intel/compilers_and_libraries/mac/mkl"
    exit 1
fi

pushd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null
rm -rf build
mkdir build
pushd build > /dev/null
cp "$MKLROOT/tools/builder/makefile" .

if [ "$(uname)" = "Darwin" ] ; then 
    make libuni export=../funcs.txt name=libtensor_mkl threading=sequential "MKLROOT=$MKLROOT"
    mv libtensor_mkl.dylib ..
elif [ "$(uname)" = "Linux" ] ; then
    make libintel64 export=../funcs.txt name=libtensor_mkl manifest=no parallel=gnu "MKLROOT=$MKLROOT"
    mv libtensor_mkl.so ..
else
    echo "Unsupported platform: $(uname)"
    exit 1
fi

popd > /dev/null
rm -rf build
popd > /dev/null
