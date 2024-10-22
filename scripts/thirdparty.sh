#!/bin/bash

# cd to the script dir
cd $(dirname $0)

# cd to the directory of /thirdparty/build
mkdir -p ../thirdparty/build
cd ../thirdparty/build

# cmake
cmake .. -G Ninja

# build
ninja -j$(nproc)
