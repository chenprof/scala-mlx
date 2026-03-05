#!/usr/bin/env bash
# setup.sh — build mlx-c, compile the C glue layer, download the model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MLX_C_DIR="$(cd "$REPO_DIR/../mlx-c" && pwd)"
BUILD_DIR="$REPO_DIR/.build"
INSTALL_DIR="$BUILD_DIR/install"

# ---- 1. Prerequisites -------------------------------------------------------

if ! command -v cmake &>/dev/null; then
  echo "==> Installing cmake..."
  brew install cmake
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "==> Installing huggingface_hub..."
  pip3 install -q huggingface_hub
fi

# ---- 2. Build mlx-c as a shared dylib ---------------------------------------

echo "==> Building mlx-c from $MLX_C_DIR ..."
cmake -S "$MLX_C_DIR" -B "$BUILD_DIR/mlxc" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

cmake --build "$BUILD_DIR/mlxc" --parallel "$(sysctl -n hw.ncpu)"
cmake --install "$BUILD_DIR/mlxc"

echo "==> mlx-c installed to $INSTALL_DIR"

# ---- 3. Compile C + C++ glue layer into libmlxllm.dylib --------------------
# .csrc/ lives in a hidden directory so Scala Native never picks it up
# during its recursive C-file scan — it just links against the dylib.
#
# mlx_llm.c      — C glue: model load, prefill, decode step, KV cache
# mlx_llm_cpp.cpp — C++ native decode loop: GPU sampling, double-buffered pipeline

echo "==> Compiling .csrc/mlx_llm.c ..."
clang -c -O2 -std=c17 \
  -I"$INSTALL_DIR/include" \
  -I"$REPO_DIR/.csrc" \
  -o "$BUILD_DIR/mlx_llm.o" \
  "$REPO_DIR/.csrc/mlx_llm.c"

echo "==> Compiling .csrc/mlx_llm_cpp.cpp ..."
clang++ -c -O2 -std=c++17 \
  -I"$INSTALL_DIR/include" \
  -I"$REPO_DIR/.csrc" \
  -o "$BUILD_DIR/mlx_llm_cpp.o" \
  "$REPO_DIR/.csrc/mlx_llm_cpp.cpp"

echo "==> Linking .build/libmlxllm.dylib ..."
clang++ -dynamiclib \
  -o "$BUILD_DIR/libmlxllm.dylib" \
  "$BUILD_DIR/mlx_llm.o" \
  "$BUILD_DIR/mlx_llm_cpp.o" \
  -L"$INSTALL_DIR/lib" -lmlxc -lmlx -lc++ \
  -Wl,-rpath,@loader_path

echo "==> $BUILD_DIR/libmlxllm.dylib ready"

# ---- 4. Download Qwen3-0.6B-4bit model --------------------------------------

MODEL_DIR="$REPO_DIR/model"
mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "==> Model already present, skipping download."
else
  echo "==> Downloading mlx-community/Qwen3-0.6B-4bit (~335 MB) ..."
  huggingface-cli download mlx-community/Qwen3-0.6B-4bit \
    config.json tokenizer.json tokenizer_config.json model.safetensors \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False
fi

echo ""
echo "======================================================"
echo "  Setup complete!"
echo "  Run the demo:"
echo "    ./test-scala-mlx.sh \"Write a haiku about Apple Silicon\""
echo "======================================================"
