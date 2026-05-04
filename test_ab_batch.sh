#!/bin/bash
# A/B test: baseline vs optimized daemon for batched TTS requests.
#
# Sends 5 chunks of the same text to the daemon via stdin pipelining
# (all at once, simulating concurrent HTTP requests), and measures
# total wall time from first command written to last DONE received.

MODEL_DIR="/mnt/data/models_storage/TTS/base"
MODEL_NAME="qwen3-tts-1.7b-f16.gguf"
EMBED="/mnt/data/models_storage/TTS/voices/.cache/donald-trump_qwen3-tts-1.7b-f16.gguf_af2dd330bf1dfb46.json"
OUTDIR="/tmp/tts_ab_test"

BASELINE_BIN="./build/qwen3-tts-cli-baseline"
OPTIMIZED_BIN="./build/qwen3-tts.cpp/qwen3-tts-cli"

# 5 text chunks of varying lengths (simulates frontend pre-chunking)
CHUNKS=(
  "Ladies and gentlemen, we are going to make this country great again, believe me. Nobody knows the system better than me."
  "The economy is doing fantastically well, the stock market is at record highs."
  "We have built the greatest military in the history of our country, and we are taking care of our incredible veterans like never before, that I can tell you."
  "The fake news media doesn't want to report the truth, they never do, but the American people are smart, very smart."
  "Thank you all, God bless you, and God bless the United States of America."
)

run_test() {
    local label="$1"
    local binary="$2"
    local outdir="${OUTDIR}/${label}"
    
    rm -rf "$outdir"
    mkdir -p "$outdir"
    
    echo "=== Starting ${label} test ==="
    echo "Binary: ${binary}"
    
    # Build the batch input: all 5 commands separated by newlines
    local input=""
    for i in "${!CHUNKS[@]}"; do
        local wav="${outdir}/chunk_${i}.wav"
        # Format: TEXT|OUTPUT|SPEAKER|REF|INSTRUCT|EMBEDDING
        input+="${CHUNKS[$i]}|${wav}||||${EMBED}\n"
    done
    
    # Start daemon and time the entire batch
    local start_time=$(date +%s%N)
    
    # Send all commands at once (simulates pipelined stdin from concurrent HTTP reqs),
    # then close stdin to let daemon exit. Capture stdout for DONE/ERROR parsing.
    local output
    output=$(printf "$input" | timeout 120 "$binary" -m "$MODEL_DIR" --model-name "$MODEL_NAME" --daemon 2>/dev/null)
    local exit_code=$?
    
    local end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))
    
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 124 ]; then
        echo "ERROR: daemon exited with code $exit_code"
        return 1
    fi
    
    # Count successful DONEs
    local done_count=$(echo "$output" | grep -c "^DONE|")
    local error_count=$(echo "$output" | grep -c "^ERROR|")
    
    # Calculate total audio duration
    local total_audio_sec=0
    for i in "${!CHUNKS[@]}"; do
        local wav="${outdir}/chunk_${i}.wav"
        if [ -f "$wav" ]; then
            local dur=$(soxi -D "$wav" 2>/dev/null || echo "0")
            total_audio_sec=$(echo "$total_audio_sec + $dur" | bc 2>/dev/null || echo "$total_audio_sec")
        fi
    done
    
    local elapsed_sec=$(echo "scale=2; $elapsed_ms / 1000" | bc)
    local rtf="N/A"
    if [ "$total_audio_sec" != "0" ] && [ "$total_audio_sec" != "" ]; then
        rtf=$(echo "scale=3; $elapsed_sec / $total_audio_sec" | bc)
    fi
    
    echo ""
    echo "Results for ${label}:"
    echo "  Chunks sent:      ${#CHUNKS[@]}"
    echo "  DONEs received:   ${done_count}"
    echo "  Errors:           ${error_count}"
    echo "  Total wall time:  ${elapsed_sec}s"
    echo "  Total audio:      ${total_audio_sec}s"
    echo "  Batch RTF:        ${rtf}"
    echo "  Output files:"
    for i in "${!CHUNKS[@]}"; do
        local wav="${outdir}/chunk_${i}.wav"
        if [ -f "$wav" ]; then
            local size=$(stat -c%s "$wav" 2>/dev/null || echo "?")
            echo "    chunk_${i}.wav: ${size} bytes"
        else
            echo "    chunk_${i}.wav: MISSING"
        fi
    done
    echo ""
    
    # Return elapsed_ms for comparison
    eval "${label}_elapsed=$elapsed_ms"
    eval "${label}_audio=$total_audio_sec"
}

echo "================================================================"
echo "  A/B Batch Performance Test: Baseline vs Optimized"
echo "  Model: ${MODEL_NAME}"
echo "  Voice: donald-trump (pre-cached embedding)"
echo "  Chunks: ${#CHUNKS[@]}"
echo "================================================================"
echo ""

# Check binaries exist
if [ ! -x "$BASELINE_BIN" ]; then
    echo "WARNING: Baseline binary not found at $BASELINE_BIN"
    echo "Skipping baseline test."
    SKIP_BASELINE=1
fi
if [ ! -x "$OPTIMIZED_BIN" ]; then
    echo "ERROR: Optimized binary not found at $OPTIMIZED_BIN"
    exit 1
fi

# Run baseline
if [ -z "$SKIP_BASELINE" ]; then
    run_test "baseline" "$BASELINE_BIN"
fi

# Small cooldown
echo "--- Cooldown 3s ---"
sleep 3

# Run optimized
run_test "optimized" "$OPTIMIZED_BIN"

# Summary comparison
echo "================================================================"
echo "  COMPARISON"
echo "================================================================"
if [ -z "$SKIP_BASELINE" ]; then
    echo "  Baseline:   ${baseline_elapsed}ms (RTF=$(echo "scale=3; ${baseline_elapsed}/1000/${baseline_audio}" | bc 2>/dev/null))"
fi
echo "  Optimized:  ${optimized_elapsed}ms (RTF=$(echo "scale=3; ${optimized_elapsed}/1000/${optimized_audio}" | bc 2>/dev/null))"
if [ -z "$SKIP_BASELINE" ] && [ "$baseline_elapsed" -gt 0 ]; then
    local speedup=$(echo "scale=1; ($baseline_elapsed - $optimized_elapsed) * 100 / $baseline_elapsed" | bc 2>/dev/null)
    echo "  Speedup:    ${speedup}% faster"
fi
echo "================================================================"
