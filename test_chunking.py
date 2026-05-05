#!/usr/bin/env python3
"""Test suite for TTS chunking.

Validates normalizer.chunk_for_tts() — model-capacity-based chunking
with newline preservation and paragraph-aware splitting.

Usage:
    cd /home/haervwe/LLMS/qwen-tts && source .venv/bin/activate && python3 test_chunking.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalizer import normalizer, MAX_CHUNK_CHARS

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}✓{RESET} {name}")
        passed += 1
    else:
        print(f"  {RED}✗{RESET} {name}: {detail}")
        failed += 1

def show_chunks(chunks):
    for i, c in enumerate(chunks):
        # Show newlines visually
        display = c.replace('\n', '\\n')
        print(f"      [{i}] ({len(c)}c) {display[:120]}{'…' if len(display) > 120 else ''}")
    print()

def run_tests():
    global passed, failed

    # --- 1. Short text → single chunk, no splitting ---
    print(f"{BOLD}1. Short text passes through as-is{RESET}")
    chunks = normalizer.chunk_for_tts("Hello, how are you?", lang="en")
    check("Single sentence → 1 chunk", len(chunks) == 1, f"got {len(chunks)}")
    show_chunks(chunks)

    # --- 2. Multiple sentences stay together (well under capacity) ---
    print(f"{BOLD}2. Multiple sentences stay in one chunk{RESET}")
    text = "Hello! How are you? I am fine. Thanks for asking."
    chunks = normalizer.chunk_for_tts(text, lang="en")
    check("4 sentences (49c) → 1 chunk", len(chunks) == 1, f"got {len(chunks)}")
    show_chunks(chunks)

    # --- 3. Paragraphs with newlines stay together when under capacity ---
    print(f"{BOLD}3. Paragraphs stay together when under capacity{RESET}")
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    chunks = normalizer.chunk_for_tts(text, lang="en")
    check("3 short paragraphs → 1 chunk", len(chunks) == 1, f"got {len(chunks)}")
    # Check that \n\n is preserved inside the chunk
    check("Double-newlines preserved", '\n\n' in chunks[0], f"chunk: {repr(chunks[0][:80])}")
    show_chunks(chunks)

    # --- 4. Single newlines preserved within text ---
    print(f"{BOLD}4. Single newlines preserved{RESET}")
    text = "Line one.\nLine two.\nLine three."
    chunks = normalizer.chunk_for_tts(text, lang="en")
    check("Single newlines stay in chunk", '\n' in chunks[0], f"chunk: {repr(chunks[0])}")
    show_chunks(chunks)

    # --- 5. Text exceeding capacity splits at paragraph boundaries ---
    print(f"{BOLD}5. Over-capacity text splits at paragraphs{RESET}")
    # Build text that exceeds MAX_CHUNK_CHARS with clear paragraph boundaries
    para = "The quick brown fox jumps over the lazy dog. " * 10  # ~450 chars
    text = (para.strip() + "\n\n") * 8  # ~3600 chars total, > 3000
    text = text.strip()
    chunks = normalizer.chunk_for_tts(text, lang="en")
    check(f"Split into 2+ chunks", len(chunks) >= 2, f"got {len(chunks)}")
    for i, c in enumerate(chunks):
        check(f"Chunk {i} under capacity", len(c) <= MAX_CHUNK_CHARS, f"{len(c)}c > {MAX_CHUNK_CHARS}")
    show_chunks(chunks)

    # --- 6. Terminal punctuation enforcement ---
    print(f"{BOLD}6. Terminal punctuation{RESET}")
    chunks = normalizer.chunk_for_tts("Text without period", lang="en")
    check("Period added", chunks[0].endswith('.'), f"ends with: {chunks[0][-5:]}")
    
    chunks2 = normalizer.chunk_for_tts("Already has period.", lang="en")
    check("No double period", not chunks2[0].endswith('..'), f"chunk: {chunks2[0]}")
    
    chunks3 = normalizer.chunk_for_tts("Question?", lang="en")
    check("Question mark preserved", chunks3[0].endswith('?'), f"chunk: {chunks3[0]}")
    print()

    # --- 7. Edge cases ---
    print(f"{BOLD}7. Edge cases{RESET}")
    check("Empty string → []", normalizer.chunk_for_tts("", lang="en") == [])
    check("Whitespace → []", normalizer.chunk_for_tts("   ", lang="en") == [])
    check("Only newlines → []", normalizer.chunk_for_tts("\n\n\n", lang="en") == [])
    single = normalizer.chunk_for_tts("Hi", lang="en")
    check("Single word → 1 chunk", len(single) == 1 and single[0] == "Hi.", f"got: {single}")
    print()

    # --- 8. Punctuation hygiene ---
    print(f"{BOLD}8. Punctuation hygiene{RESET}")
    processed, _ = normalizer.process("He arrived—tired and hungry—at the station")
    check("Em-dash → comma", '—' not in processed and ', ' in processed, f"got: {processed}")
    
    processed2, _ = normalizer.process("Wait...... here")
    check("Ellipsis normalized", '......' not in processed2, f"got: {processed2}")
    
    processed3, _ = normalizer.process("Really!!! Amazing!!")
    check("Repeated punct deduped", '!!!' not in processed3, f"got: {processed3}")
    print()

    # --- 9. Newlines survive process() ---
    print(f"{BOLD}9. Newlines survive normalization{RESET}")
    processed4, _ = normalizer.process("First paragraph.\n\nSecond paragraph.")
    check("Double newline preserved through process()", '\n\n' in processed4, f"got: {repr(processed4)}")
    
    processed5, _ = normalizer.process("Line one.\nLine two.")
    check("Single newline preserved through process()", '\n' in processed5, f"got: {repr(processed5)}")
    print()

    # --- 10. Sentence-level fallback for massive single paragraph ---
    print(f"{BOLD}10. Massive paragraph splits at sentences{RESET}")
    # Single paragraph with no \n\n, exceeding capacity
    big_para = ("This is a sentence about something. " * 100).strip()
    chunks = normalizer.chunk_for_tts(big_para, lang="en")
    check(f"Massive {len(big_para)}c paragraph splits", len(chunks) >= 2, f"got {len(chunks)}")
    for i, c in enumerate(chunks):
        check(f"Chunk {i} under capacity", len(c) <= MAX_CHUNK_CHARS, f"{len(c)}c > {MAX_CHUNK_CHARS}")
    show_chunks(chunks)

    # --- Summary ---
    print(f"{BOLD}{'='*50}{RESET}")
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}{BOLD}All {total} tests passed!{RESET}")
    else:
        print(f"{RED}{BOLD}{failed}/{total} tests FAILED{RESET}")
    print()
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
