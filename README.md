# Qwen3-TTS ROCm API Server

An OpenAI-compatible Text-to-Speech (TTS) server built on the `qwen3-tts.cpp` backend, optimized for AMD GPUs (ROCm).

## Features

- **OpenAI Compatible**: Supports `/v1/audio/speech` and `/v1/models`.
- **ROCm Support**: Hardware acceleration for AMD GPUs.
- **Voice Cloning**: High-quality cloning for Base models using 5-10s reference WAVs.
- **Style Steering**: Natural language steering (e.g., "whispering") for 1.7B models.
- **Preset System**: Combine voices and styles into reusable presets.
- **Dynamic Discovery**: Automatically detects model capabilities and speakers.

## Directory Structure

- `/mnt/data/models_storage/TTS/base`: GGUF model files.
- `/mnt/data/models_storage/TTS/voices`: Reference `.wav` files for cloning.
- `presets.json`: Custom combinations of voices and instructions.

## Usage

### Listing Models
```bash
curl http://localhost:8001/v1/models
```

### Listing Voices (including Presets)
```bash
curl http://localhost:8001/v1/audio/voices?model=qwen3-tts-1.7b-customvoice-f16.gguf
```

### Generating Speech (Cloning)
Works with `0.6b-f16.gguf` or `1.7b-f16.gguf`.
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-0.6b-f16.gguf",
    "input": "Hello, world!",
    "voice": "marta"
  }' --output output.wav
```

### Generating Speech (Style Steering)
Works with `1.7b-customvoice-f16.gguf`.
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-1.7b-customvoice-f16.gguf",
    "input": "I am whispering right now.",
    "voice": "vivian",
    "instruction": "Whispering, very soft and quiet voice."
  }' --output output.wav
```

### Using Presets
You can define presets in `presets.json`. For example, selecting `whispering-vivian` will automatically use the `vivian` speaker and apply the whispering instruction.
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-1.7b-customvoice-f16.gguf",
    "input": "I am a preset.",
    "voice": "whispering-vivian"
  }' --output output.wav
```

## Configuration

### Adding Presets
Edit `presets.json` in the root directory:
```json
{
  "my-preset": {
    "voice": "vivian",
    "instruction": "Speak very fast and high pitched."
  }
}
```

### Adding Voice Clones
Place a 5-10 second WAV file in `/mnt/data/models_storage/TTS/voices/` and refer to it by its filename (without `.wav`) in the `voice` parameter.
