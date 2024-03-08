#!/bin/bash

set -xe

# Step 0: Install required packages.
python3 -m pip install -U --no-cache-dir pip setuptools cython cmake torch_runstats numpy pandas
apt-get update -y
apt-get install -y zip gfortran libgtest-dev libopenblas-dev libfftw3-dev libfftw3-double3 libfftw3-single3 libfftw3-3 libfftw3-bin

# Define working directory
WORK_DIR=$(dirname "$(readlink -f "$0")")
cd ${WORK_DIR}

# Determine PyTorch and CUDA versions
PYTORCH_VERSION="2.1.0"
CUDA_VERSION="12.1"

# Define libtorch download URL
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcu${CUDA_VERSION}.zip"
echo "Libtorch url: ${LIBTORCH_URL}"

# Download and extract libtorch if not already installed
TORCH_CMAKE='/opt/libtorch/share/cmake/Torch/TorchConfig.cmake'
if [ ! -f "${TORCH_CMAKE}" ]; then
    wget -O libtorch.zip "${LIBTORCH_URL}" && unzip libtorch.zip -d /opt && rm libtorch.zip
    echo "libtorch downloaded and extracted to /opt/libtorch."
else
    echo "libtorch is already installed in /opt/libtorch."
fi

# Clone LAMMPS repository if CMakeLists.txt is not found
CMAKE_PATH="./lammps/cmake/CMakeLists.txt"
if [ ! -f "${CMAKE_PATH}" ]; then
    git clone --depth 1 https://github.com/lammps/lammps.git -b stable_2Aug2023_update3
    echo "LAMMPS cloned."
else
    echo "LAMMPS CMakeLists.txt found. Skipping clone."
fi

if [ ! -f "${CMAKE_PATH}" ]; then
    echo "Clone failed."
    exit 1
fi

# Update build.sh in lammps directory
cp -u ./build.sh ./lammps/

# Copy custom pair files into LAMMPS src directory if not already present
for file in pair_bamboo.cpp pair_bamboo.h; do
    if [ ! -f "./lammps/src/${file}" ]; then
        cp "./src/${file}" ./lammps/src/
    fi
done

# Copy custom KOKKOS pair files into LAMMPS KOKKOS directory if not already present
for file in pair_bamboo_kokkos.cpp pair_bamboo_kokkos.h; do
    if [ ! -f "./lammps/src/KOKKOS/${file}" ]; then
        cp "./src/${file}" ./lammps/src/KOKKOS/
    fi
done

# Append Torch configuration to CMakeLists.txt if not already done
if ! grep -q "find_package(Torch REQUIRED)" "$CMAKE_PATH"; then
    cat >> "$CMAKE_PATH" << EOF

# Find the Torch package
find_package(Torch REQUIRED)

# Add the Torch CXX flags to the compilation options
set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} \${TORCH_CXX_FLAGS}")

# Link the target against the Torch libraries
target_link_libraries(lammps PUBLIC "\${TORCH_LIBRARIES}")
EOF
    echo "Torch configuration appended."
else
    echo "Torch configuration already present."
fi
