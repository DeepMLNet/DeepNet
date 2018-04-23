#!/usr/bin/env bash 

set -e

if [ -z "$MKLROOT" ]; then
    echo "Set the MKLROOT environment variable to the root of your Intel MKL directory,"
    echo "for example:"
    echo "             Linux: export MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl"
    echo "             Mac:   export MKLROOT=/opt/mac/compilers_and_libraries/linux/mkl"
    exit 1
fi

pushd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null
rm -rf build
mkdir build
pushd build > /dev/null
cp "$MKLROOT/tools/builder/makefile" .

if [ "$(uname)" = "Darwin" ] ; then 
    TARGET="libuni"
elif [ "$(uname)" = "Linux" ] ; then
    TARGET="libintel64"
else
    echo "Unsupported platform: $(uname)"
    exit 1
fi

make $TARGET export=../funcs.txt name=../libtensor_mkl manifest=no parallel=gnu "MKLROOT=$MKLROOT"

popd > /dev/null
rm -rf build
popd > /dev/null




