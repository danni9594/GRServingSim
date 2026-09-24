#!/usr/bin/env bash
# Build only the backend/schema needed by GR; requires CMake, protoc, libprotobuf.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
git submodule update --init astra-sim
git -C astra-sim submodule update --init \
  extern/graph_frontend/chakra extern/helper/fmt extern/helper/spdlog \
  extern/memory_backend/analytical extern/network_backend/analytical
git -C astra-sim/extern/network_backend/analytical submodule update --init extern/yaml-cpp
bash scripts/apply-gr-patches.sh
PROTO_DIR="$REPO_ROOT/astra-sim/extern/graph_frontend/chakra/schema/protobuf"
protoc --proto_path="$PROTO_DIR" --cpp_out="$PROTO_DIR" --python_out="$PROTO_DIR" et_def.proto
BUILD_DIR="$REPO_ROOT/astra-sim/build/astra_analytical/build"
cmake -S astra-sim/build/astra_analytical -B "$BUILD_DIR" \
  -DBUILDTARGET=congestion_unaware -DCMAKE_BUILD_TYPE=Release -DYAML_CPP_BUILD_TESTS=OFF
cmake --build "$BUILD_DIR" -j "${GR_BUILD_JOBS:-4}"
# Legacy LLM frontend expects this compatibility path.
mkdir -p "$BUILD_DIR/AnalyticalAstra/bin"
ln -sfn ../../bin/AstraSim_Analytical_Congestion_Unaware "$BUILD_DIR/AnalyticalAstra/bin/AnalyticalAstra"
