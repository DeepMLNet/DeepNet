#!/usr/bin/env bash 

set -e

if [ -z "$MKLROOT" ]; then
    echo "Set the MKLROOT environment variable to the root of your Intel MKL directory,"
    echo "for example: export MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl"
    exit 1
fi

pushd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null
rm -rf build
mkdir build
pushd build > /dev/null
cp "$MKLROOT/tools/builder/makefile" .

make libintel64 export=../funcs.txt name=../libtensor_mkl manifest=no parallel=gnu "MKLROOT=$MKLROOT"

popd > /dev/null
rm -rf build
popd > /dev/null




