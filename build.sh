#!/usr/bin/env bash
# Build DREAMPlace inside the CUDA 11.8 dependency image.
# Invoke from external/DREAMPlace via:
#   docker run --rm --gpus all -v ${PWD}:/dreamplace -w /dreamplace bowrango/dreamplace:cuda118 bash build.sh
#
# By default this lets DREAMPlace's CMakeLists.txt choose CUDA architectures.
# To force a list, pass e.g.
#   -e DREAMPLACE_CUDA_ARCHITECTURES='8.6;8.9'
#
# --gpus all is REQUIRED at build time, not just runtime. DREAMPlace's
# cmake/TorchExtension.cmake checks torch.cuda.is_available() at configure
# time; without a GPU attached, TORCH_ENABLE_CUDA=0 and CUDA kernels are
# silently skipped from the build, producing a CPU-only install.
set -euo pipefail

mkdir -p build install
cd build

cmake_args=(
  ..
  -DCMAKE_INSTALL_PREFIX=/dreamplace/install
  -DPython_EXECUTABLE="$(which python3)"
)

if [[ -n "${DREAMPLACE_CUDA_ARCHITECTURES:-}" ]]; then
  cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${DREAMPLACE_CUDA_ARCHITECTURES}")
fi

cmake "${cmake_args[@]}"
make -j8
make install
