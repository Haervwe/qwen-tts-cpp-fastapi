import asyncio
import os
import uuid
import json
import subprocess
import hashlib
import re
import wave
import io
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

app = FastAPI(title="Qwen3-TTS OpenAI-Compatible API")

MODELS_BASE_DIR = Path("/mnt/data/models_storage/TTS/base")
VOICES_DIR = Path("/mnt/data/models_storage/TTS/voices")
EMBED_CACHE_DIR = VOICES_DIR / ".cache"
CLI_PATH = Path("/home/haervwe/LLMS/qwen-tts/build/qwen3-tts.cpp/qwen3-tts-cli")

# Ensure cache directory exists
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default model if not specified
DEFAULT_MODEL = "qwen3-tts-0.6b-f16.gguf"

# Cache for model info and speakers
_model_cache = {}

def get_model_capabilities(model_name: str):
    if not model_name.endswith(".gguf"):
        model_name += ".gguf"
        
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    info = {"speakers": [], "model_type": "base", "supports_voice_clone": False}
    
    # Get basic info
    cmd_info = [str(CLI_PATH), "-m", str(MODELS_BASE_DIR), "--model-name", model_name, "--info"]
    try:
        res = subprocess.run(cmd_info, capture_output=True, text=True, timeout=10)
        output = res.stdout
        start = output.find("{")
        if start != -1:
            info.update(json.loads(output[start:]))
    except Exception as e:
        print(f"Error getting model info: {e}")

    # Get speakers list
    cmd_speakers = [str(CLI_PATH), "-m", str(MODELS_BASE_DIR), "--model-name", model_name, "--list-speakers"]
    try:
        res = subprocess.run(cmd_speakers, capture_output=True, text=True, timeout=10)
        output = res.stdout
        start = output.find("[")
        if start != -1:
            info["speakers"] = json.loads(output[start:])
    except Exception as e:
        print(f"Error getting speakers list: {e}")
        
    _model_cache[model_name] = info
    return info

class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = "default"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    extra_body: Optional[dict] = {}

# Load presets from presets.json
PRESETS_FILE = Path("/home/haervwe/LLMS/qwen-tts/presets.json")

def load_presets():
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading presets: {e}")
    return {}

@app.get("/v1/models")
async def list_models():
    models_data = []
    # Add actual model files
    for f in MODELS_BASE_DIR.glob("*.gguf"):
        models_data.append({
            "id": f.name,
            "object": "model",
            "created": int(f.stat().st_mtime),
            "owned_by": "qwen"
        })
    
    # Also add voices as models for better integration with generic clients
    voices_resp = await list_voices()
    for v in voices_resp.get("data", []):
        models_data.append({
            "id": v["id"],
            "object": "model",
            "created": 1600000000,
            "owned_by": "system"
        })
        
    return {"object": "list", "data": models_data}

@app.get("/v1/audio/voices")
@app.get("/v1/voices")
async def list_voices(model: Optional[str] = None):
    # Reload presets to catch changes
    current_presets = load_presets()
            
    # Gather all available voices
    voice_files = [f.stem for f in VOICES_DIR.glob("*.wav")]
    preset_names = list(current_presets.keys())
    
    # Also include internal speakers for the requested model or default
    target_model = model or DEFAULT_MODEL
    info = get_model_capabilities(target_model)
    internal_speakers = info.get("speakers", []) if info else []
    model_type = info.get("model_type", "base") if info else "base"
    
    # Return structured data
    voices_data = []
    
    # 1. Internal speakers
    for name in internal_speakers:
        voices_data.append({
            "id": name,
            "voice_id": name,
            "name": name.title(),
            "label": name.title(),
            "value": name,
            "category": "internal"
        })
        
    # 2. Presets
    for name in preset_names:
        voices_data.append({
            "id": name,
            "voice_id": name,
            "name": name.replace('_', ' ').title(),
            "label": name.replace('_', ' ').title(),
            "value": name,
            "category": "preset"
        })

    # 3. WAV Clones (ONLY for base models)
    if model_type == "base":
        for name in voice_files:
            if name not in internal_speakers and name not in preset_names:
                voices_data.append({
                    "id": name,
                    "voice_id": name,
                    "name": name.replace('_', ' ').title(),
                    "label": name.replace('_', ' ').title(),
                    "value": name,
                    "category": "cloned"
                })
    
    # Returning a dictionary with both 'voices' (strings) and 'data' (objects) 
    # covers most UI implementations (OpenWebUI, ElevenLabs-style, etc.)
    # and should resolve '[object Object]' rendering issues.
    return {
        "voices": [v["id"] for v in voices_data],
        "data": voices_data,
        "object": "list"
    }

def get_file_sha256(file_path: Path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return "unknown"

def normalize_text(text: str) -> str:
    """Basic text normalization for TTS."""
    abbreviations = {
        r"\bDr\.\b": "Doctor",
        r"\bMr\.\b": "Mister",
        r"\bMs\.\b": "Miss",
        r"\bMrs\.\b": "Missus",
        r"\bSt\.\b": "Street",
        r"\bAve\.\b": "Avenue",
        r"\bRd\.\b": "Road",
        r"\bvs\.\b": "versus",
        r"\betc\.\b": "et cetera",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

managers: Dict[str, 'ModelManager'] = {}
manager_lock = asyncio.Lock()

class ModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.proc = None
        self.lock = asyncio.Lock()
        self.stderr_task = None

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return

        info = get_model_capabilities(self.model_name)
        model_path = info.get("path", str(MODELS_BASE_DIR))
        model_file = info.get("model_name", self.model_name)
        
        cmd = [
            str(CLI_PATH),
            "-m", model_path,
            "--model-name", model_file,
            "--daemon"
        ]
        
        print(f"Starting model daemon: {' '.join(cmd)}")
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for READY
        line = await self.proc.stdout.readline()
        if b"READY" not in line:
            stderr_err = await self.proc.stderr.read(1024)
            raise Exception(f"Model failed to start: {line.decode().strip()} - {stderr_err.decode()}")
            
        print(f"Model {self.model_name} is READY")
        self.stderr_task = asyncio.create_task(self._log_stderr())

    async def _log_stderr(self):
        try:
            while self.proc and self.proc.returncode is None:
                line = await self.proc.stderr.readline()
                if not line: break
                print(f"[{self.model_name}] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass

    async def synthesize(self, text, output, speaker="", reference="", instruct="", embedding="", retry=True):
        async with self.lock:
            try:
                if not self.proc or self.proc.returncode is not None:
                    await self.start()
                
                # Format: TEXT|OUTPUT|SPEAKER|REF|INSTRUCT|EMBEDDING
                cmd_line = f"{text}|{output}|{speaker}|{reference}|{instruct}|{embedding}\n"
                self.proc.stdin.write(cmd_line.encode())
                await self.proc.stdin.drain()
                
                line = await self.proc.stdout.readline()
                if not line:
                    if retry:
                        print(f"Model daemon died, restarting and retrying...")
                        self.proc = None
                        return await self.synthesize(text, output, speaker, reference, instruct, embedding, retry=False)
                    return False, "Model process terminated unexpectedly"
                    
                resp = line.decode().strip()
                if resp.startswith("DONE|"):
                    return True, resp.split("|")[1]
                elif resp.startswith("ERROR|"):
                    return False, resp.split("|")[1]
                else:
                    return False, f"Unexpected response: {resp}"
            except Exception as e:
                if retry:
                    print(f"Synthesis error: {e}, retrying...")
                    self.proc = None
                    return await self.synthesize(text, output, speaker, reference, instruct, embedding, retry=False)
                return False, str(e)

async def get_manager(model_name: str) -> ModelManager:
    async with manager_lock:
        if model_name not in managers:
            managers[model_name] = ModelManager(model_name)
        manager = managers[model_name]
    await manager.start()
    return manager

def split_text(text: str, max_chars: int = 300) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk: chunks.append(current_chunk)
            if len(sentence) > max_chars:
                sub_sentences = re.split(r'(?<=[,;])\s+', sentence)
                temp_chunk = ""
                for sub in sub_sentences:
                    if len(temp_chunk) + len(sub) < max_chars:
                        temp_chunk += (" " if temp_chunk else "") + sub
                    else:
                        if temp_chunk: chunks.append(temp_chunk)
                        temp_chunk = sub
                current_chunk = temp_chunk
            else:
                current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    return chunks

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    model_name = req.model or DEFAULT_MODEL
    text = normalize_text(req.input)
    voice_name = req.voice
    instruction = req.extra_body.get("instruction") if req.extra_body else None

    info = get_model_capabilities(model_name)
    current_presets = load_presets()
    if voice_name in current_presets:
        preset = current_presets[voice_name]
        voice_name = preset.get("voice", voice_name)
        if not instruction:
            instruction = preset.get("instruction")

    text_chunks = split_text(text)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="Empty input text")

    request_id = uuid.uuid4().hex
    manager = await get_manager(model_name)
    
    # Determine voice params once for all chunks
    speaker = ""
    reference = ""
    embedding = ""
    
    if info and voice_name in info.get("speakers", []):
        speaker = voice_name
    else:
        # Voice cloning is ONLY for base models
        model_type = info.get("model_type", "base")
        reference_wav = VOICES_DIR / f"{voice_name}.wav"
        
        if model_type == "base" and reference_wav.exists() and info.get("supports_voice_clone"):
            wav_hash = get_file_sha256(reference_wav)
            # Use full model name to differentiate dimensions (0.6b vs 1.7b)
            embed_cache_path = EMBED_CACHE_DIR / f"{voice_name}_{model_name}_{wav_hash}.json"
            
            if embed_cache_path.exists():
                embedding = str(embed_cache_path)
            else:
                # Extract embedding once for the first chunk and save to cache
                print(f"Extracting new embedding for {voice_name} using {model_name}...")
                reference = str(reference_wav)
                embedding = str(embed_cache_path) 
        else:
            # Fallback for non-base models or missing WAVs
            if info and info.get("speakers"):
                # Use the first available internal speaker as fallback
                speaker = info.get("speakers")[0]
            else:
                speaker = "vivian"

    temp_wavs = []
    try:
        for i, chunk in enumerate(text_chunks):
            chunk_wav = Path(f"/tmp/chunk_{request_id}_{i}.wav")
            
            # For the first chunk, if we need to extract, we pass both reference and the target embedding path
            # For subsequent chunks, we only pass the embedding path
            current_reference = reference if i == 0 else ""
            
            success, result_path = await manager.synthesize(
                text=chunk,
                output=str(chunk_wav),
                speaker=speaker,
                reference=current_reference,
                instruct=instruction or "",
                embedding=embedding # This will be the cache path
            )

            if success and chunk_wav.exists():
                temp_wavs.append(chunk_wav)
            else:
                print(f"Error synthesizing chunk {i}: {result_path}")

        if not temp_wavs:
            raise HTTPException(status_code=500, detail="Audio generation failed")

        # Concatenate into final output
        final_wav = Path(f"/tmp/final_{request_id}.wav")
        
        # Use wave module for simple concatenation
        data = []
        params = None
        for wav_path in temp_wavs:
            with wave.open(str(wav_path), 'rb') as w:
                if params is None:
                    params = w.getparams()
                data.append(w.readframes(w.getnframes()))
        
        with wave.open(str(final_wav), 'wb') as w:
            w.setparams(params)
            for d in data:
                w.writeframes(d)

        # Convert to requested format if needed (OpenAI default is mp3)
        fmt = req.response_format or "mp3"
        if fmt == "mp3":
            final_output = Path(f"/tmp/final_{request_id}.mp3")
            conv_cmd = ["ffmpeg", "-i", str(final_wav), "-codec:a", "libmp3lame", "-qscale:a", "2", str(final_output), "-y"]
            subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            media_type = "audio/mpeg"
        else:
            final_output = final_wav
            media_type = "audio/wav"

        def cleanup():
            for f in temp_wavs:
                if f.exists(): os.remove(f)
            if final_wav.exists(): os.remove(final_wav)
            if final_output != final_wav and final_output.exists(): os.remove(final_output)

        return FileResponse(final_output, media_type=media_type, background=BackgroundTask(cleanup))

    except Exception as e:
        for f in temp_wavs:
            if f.exists(): os.remove(f)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
