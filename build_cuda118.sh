#!/usr/bin/env bash
# Build DREAMPlace inside bowrango/dreamplace:cuda118 for sm_86 only (RTX A4000).
# Invoke from external/DREAMPlace via:
#   docker run --rm --gpus all -v ${PWD}:/DREAMPlace -w /DREAMPlace bowrango/dreamplace:cuda118 bash build_cuda118.sh
#
# --gpus all is REQUIRED at build time, not just runtime. DREAMPlace's
# cmake/TorchExtension.cmake checks torch.cuda.is_available() at configure
# time; without a GPU attached, TORCH_ENABLE_CUDA=0 and CUDA kernels are
# silently skipped from the build, producing a CPU-only install.
set -euo pipefail

mkdir -p build install
cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/DREAMPlace/install \
  -DPython_EXECUTABLE=$(which python3) \
  -DCMAKE_CUDA_ARCHITECTURES=86
make -j2
make install
