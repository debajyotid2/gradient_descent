#!/bin/sh

# Set up variables

DEP_DIR="../external"
if [ "$1" = "" ]; then
    NUM_THREADS=1
else
    NUM_THREADS=$1
fi

# Create directory for dependencies
cd $DEP_DIR || { echo "cd $DEP_DIR failed."; exit 1; }
git submodule update --init --recursive || exit 2

# Install OpenBLAS and matlibr
cd matlibr/scripts || exit 3
source install_openblas.sh "$NUM_THREADS" || exit 4
echo "Successfully installed OpenBLAS"
cd ../scripts || exit 5
source install.sh "$NUM_THREADS" || exit 6
echo "Successfully installed matlibr"
cd ../../../scripts || exit 7
