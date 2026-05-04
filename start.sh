#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

UV_PATH="/home/haervwe/.local/bin/uv"

if [ ! -d ".venv" ]; then
    $UV_PATH venv .venv
fi

source .venv/bin/activate
$UV_PATH pip install -r requirements.txt

# Ensure Whisper is setup
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    echo "Whisper CLI not found, running setup..."
    ./setup_whisper.sh
fi

# Start the server using uvicorn
# PORT can be set by llama-swap; defaults to 8001 for standalone use
exec python -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8001}"
