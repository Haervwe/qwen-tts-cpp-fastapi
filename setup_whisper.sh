#!/usr/bin/env bash
set -e

# Configuration
WHISPER_DIR="whisper.cpp"
MODEL_DIR="/mnt/data/models_storage/TTS/whisper"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"

echo "=== Setting up whisper.cpp ==="

# 1. Initialize submodules
echo "Initializing submodules..."
git submodule update --init --recursive

# 2. Fix unsupported compiler flags (hipcc/clang mismatch)
echo "Patching whisper.cpp for ROCm compatibility..."
cd "$WHISPER_DIR"
sed -i 's/-Wunreachable-code-break//g' ggml/cmake/common.cmake
sed -i 's/-Wunreachable-code-return//g' ggml/cmake/common.cmake
cd ..

# 3. Build with ROCm (HIPBLAS)
echo "Building whisper.cpp with ROCm support..."
cd "$WHISPER_DIR"
rm -rf build
mkdir -p build
cd build
cmake -DGGML_HIPBLAS=ON -DCMAKE_CXX_COMPILER=hipcc -DWHISPER_BUILD_EXAMPLES=ON ..
make -j$(nproc)
cd ../..

# 4. Download medium model
echo "Ensuring model directory exists: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/ggml-medium.bin" ]; then
    echo "Downloading Whisper medium model..."
    curl -L "$MODEL_URL" -o "$MODEL_DIR/ggml-medium.bin"
else
    echo "Whisper medium model already exists at $MODEL_DIR/ggml-medium.bin"
fi

echo "=== whisper.cpp setup complete ==="
