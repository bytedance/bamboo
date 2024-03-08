#!/bin/bash
# Exit immediately if a command exits with a non-zero status, and print each command.
set -ex

# Define the working directory based on the script's location.
WORK_DIR="$(dirname "$(readlink -f "$0")")"
cd "${WORK_DIR}"

# CUDA settings
CUDA_PATH="/usr/local/cuda"
export PATH="$PATH:${CUDA_PATH}/bin"

# Check CUDA compiler version
nvcc --version

# Define directories
BUILD_DIR="${WORK_DIR}/build"
OUT_DIR="${WORK_DIR}/output"
CMAKE_DIR="${WORK_DIR}/cmake"

# Clean up previous build and output directories, then recreate them.
rm -rf "${OUT_DIR}" && mkdir "${OUT_DIR}"
rm -rf "${BUILD_DIR}" && mkdir "${BUILD_DIR}" && cd "${BUILD_DIR}"

# CMake configuration
CMAKE_PRESETS_PATH="../cmake/presets"
CMAKE_BASIC_PRESET="${CMAKE_PRESETS_PATH}/basic.cmake"
CMAKE_KOKKOS_CUDA_PRESET="${CMAKE_PRESETS_PATH}/kokkos-cuda.cmake"

cmake "${CMAKE_DIR}" \
    -C "${CMAKE_BASIC_PRESET}" \
    -C "${CMAKE_KOKKOS_CUDA_PRESET}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_LIBRARY_PATH="${CUDA_PATH}/lib64/" \
    -DMKL_INCLUDE_DIR="/usr/include" \
    -DBUILD_TESTING=OFF \
    -DCUDAToolkit_ROOT="${CUDA_PATH}" \
    -DKokkos_ARCH_PASCAL60=OFF \
    -DKokkos_ARCH_ADA89=ON \
    -DKokkos_CUDA_DIR="${CUDA_PATH}" \
    -DKokkos_ENABLE_OPENMP=ON \
    -DPKG_GPU=yes \
    -DGPU_API=cuda \
    -DGPU_ARCH=sm_89 \
    -DTorch_DIR="/opt/libtorch/share/cmake/Torch" \
    -DBIN2C="${CUDA_PATH}/bin/bin2c" \
    -DFFT=FFTW3 \
    -DPKG_KOKKOS=ON \
    -DPKG_KSPACE=ON 

# Compile with all available cores
make -j

# Move the compiled binary to the output directory
mv ./lmp "${OUT_DIR}/lmp"

echo "Compile finished."
