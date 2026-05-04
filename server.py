import asyncio
import os
import uuid
import json
import subprocess
import hashlib
import re
import wave
import io
import logging
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from normalizer import normalizer

app = FastAPI(title="Qwen3-TTS OpenAI-Compatible API")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qwen-tts")

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).parent

MODELS_BASE_DIR = Path(os.getenv("MODELS_BASE_DIR", str(BASE_DIR / "models")))
VOICES_DIR = Path(os.getenv("VOICES_DIR", str(BASE_DIR / "voices")))
EMBED_CACHE_DIR = VOICES_DIR / ".cache"
CLI_PATH = Path(os.getenv("CLI_PATH", str(BASE_DIR / "build/qwen3-tts.cpp/qwen3-tts-cli")))

# Ensure cache directory exists
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default model if not specified
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3-tts-0.6b-f16.gguf")

# Enablement Flags (useful for llama-swap where one process = one model type)
ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() == "true"
ENABLE_STT = os.getenv("ENABLE_STT", "true").lower() == "true"

# Whisper Configuration
WHISPER_CLI_PATH = Path(os.getenv("WHISPER_CLI_PATH", str(BASE_DIR / "whisper.cpp/build/bin/whisper-cli")))
WHISPER_MODEL_PATH = Path(os.getenv("WHISPER_MODEL_PATH", "/mnt/data/models_storage/TTS/whisper/ggml-medium.bin"))
TRANSCRIPT_CACHE_DIR = EMBED_CACHE_DIR # Reuse embed cache dir for transcripts

# Cache for model info and speakers
_model_cache = {}
MODEL_CAPS_CACHE_FILE = EMBED_CACHE_DIR / "model_caps_cache.json"

def load_caps_cache():
    global _model_cache
    if MODEL_CAPS_CACHE_FILE.exists():
        try:
            with open(MODEL_CAPS_CACHE_FILE, "r") as f:
                _model_cache = json.load(f)
        except Exception as e:
            logger.error(f"Error loading caps cache: {e}")

def save_caps_cache():
    try:
        with open(MODEL_CAPS_CACHE_FILE, "w") as f:
            json.dump(_model_cache, f)
    except Exception as e:
        logger.error(f"Error saving caps cache: {e}")

# Initial load
load_caps_cache()

def get_model_capabilities(model_name: str):
    if not model_name.endswith(".gguf"):
        model_name += ".gguf"
        
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    # If TTS is disabled, don't trigger the heavy CLI probe
    if not ENABLE_TTS:
        return {"speakers": [], "model_type": "base", "supports_voice_clone": False}

    info = {"speakers": [], "model_type": "base", "supports_voice_clone": False}
    
    # Get basic info
    cmd_info = [str(CLI_PATH), "-m", str(MODELS_BASE_DIR), "--model-name", model_name, "--info"]
    try:
        logger.info(f"Probing model info: {model_name}...")
        res = subprocess.run(cmd_info, capture_output=True, text=True, timeout=10)
        output = res.stdout
        start = output.find("{")
        if start != -1:
            info.update(json.loads(output[start:]))
    except Exception as e:
        logger.error(f"Error getting model info: {e}")

    # Get speakers list
    cmd_speakers = [str(CLI_PATH), "-m", str(MODELS_BASE_DIR), "--model-name", model_name, "--list-speakers"]
    try:
        res = subprocess.run(cmd_speakers, capture_output=True, text=True, timeout=10)
        output = res.stdout
        start = output.find("[")
        if start != -1:
            info["speakers"] = json.loads(output[start:])
    except Exception as e:
        logger.error(f"Error getting speakers list: {e}")
        
    _model_cache[model_name] = info
    save_caps_cache()
    return info

class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = "default"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    extra_body: Optional[dict] = {}

class TranscriptionResponse(BaseModel):
    text: str
    task: Optional[str] = "transcribe"
    language: Optional[str] = "english"
    duration: Optional[float] = None

# Load presets from presets.json
PRESETS_FILE = Path(os.getenv("PRESETS_FILE", str(BASE_DIR / "presets.json")))

def load_presets():
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
    return {}

@app.get("/v1/models")
async def list_models():
    models_data = []
    
    # 1. Add TTS models (GGUF files)
    if ENABLE_TTS:
        for f in MODELS_BASE_DIR.glob("*.gguf"):
            models_data.append({
                "id": f.name,
                "object": "model",
                "created": int(f.stat().st_mtime),
                "owned_by": "qwen"
            })
    
    # 2. Add STT models
    if ENABLE_STT:
        models_data.append({
            "id": "whisper-1",
            "object": "model",
            "created": 1600000000,
            "owned_by": "openai"
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
    
    # 3. Internal speakers
    # Only probe for internal speakers if a model is specified OR if we want to show defaults
    # When ENABLE_TTS is false, get_model_capabilities returns empty immediately.
    target_model = model or (DEFAULT_MODEL if ENABLE_TTS else None)
    info = get_model_capabilities(target_model) if target_model else None
    
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

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: Optional[str] = "whisper-1",
    language: Optional[str] = "en",
    prompt: Optional[str] = None,
    response_format: Optional[str] = "json"
):
    """OpenAI-compatible transcription endpoint."""
    if not ENABLE_STT:
        raise HTTPException(status_code=400, detail="STT engine is disabled on this server instance.")
    
    temp_file = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    try:
        # Save upload to temp file
        with open(temp_file, "wb") as f:
            f.write(await file.read())
            
        # Transcribe
        text = await WhisperManager.transcribe(temp_file, language=language)
        
        if response_format == "text":
            return text
            
        return TranscriptionResponse(text=text, language=language)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file.exists():
            os.remove(temp_file)

def get_file_sha256(file_path: Path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return "unknown"

async def normalize_audio(input_path: Path, output_path: Path, max_duration: float = 15.0):
    """Normalize audio for ICL voice cloning: 24kHz, mono, normalized volume.
    
    Key requirements from Qwen3-TTS documentation:
    - Reference audio sweet spot is 10-15s. >20s causes generation loops/hangs.
    - Append 0.5s silence at end to prevent phoneme bleed (clicking artifacts).
    """
    # Check source duration
    probe = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    stdout, _ = await probe.communicate()
    try:
        duration = float(stdout.decode().strip())
    except (ValueError, AttributeError):
        duration = 0.0
    
    if duration > max_duration:
        # Find a silence near the cut point to truncate at a natural boundary
        search_start = max(0, max_duration - 10)
        silence_cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ss", str(search_start), "-t", "10",
            "-af", "silencedetect=noise=-30dB:d=0.3",
            "-f", "null", "-"
        ]
        silence_proc = await asyncio.create_subprocess_exec(
            *silence_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, silence_stderr = await silence_proc.communicate()
        
        import re as _re
        cut_time = max_duration
        silence_ends = _re.findall(r'silence_end: ([\d.]+)', silence_stderr.decode())
        if silence_ends:
            for ts in reversed(silence_ends):
                t = float(ts) + search_start
                if t <= max_duration and t >= max_duration - 10:
                    cut_time = t
                    break
            logger.info(f"Smart truncation: {duration:.1f}s → {cut_time:.1f}s (at silence boundary)")
        else:
            logger.info(f"No silence found near {max_duration}s, hard-cutting at {max_duration}s")
    else:
        cut_time = duration
    
    # Build ffmpeg command:
    # - Truncate to cut_time
    # - Normalize loudness
    # - Fade out at end to avoid abrupt cut  
    # - Pad 0.5s silence at end to prevent phoneme bleed (known Qwen3-TTS artifact)
    af_filters = ["loudnorm"]
    if duration > max_duration:
        af_filters.append(f"afade=t=out:st={max(0, cut_time - 0.1)}:d=0.1")
    af_filters.append("apad=pad_dur=0.5")
    
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-t", str(cut_time),
        "-ar", "24000",
        "-ac", "1",
        "-af", ",".join(af_filters),
        str(output_path),
        "-y"
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.wait()
    if proc.returncode != 0:
        _, stderr = await proc.communicate()
        logger.error(f"FFmpeg Error: {stderr.decode()}")
        raise Exception(f"ffmpeg failed to normalize audio {input_path}")

managers: Dict[str, 'ModelManager'] = {}
manager_lock = asyncio.Lock()

class WhisperManager:
    """Manages whisper-cli execution for transcription."""
    
    @staticmethod
    async def transcribe(audio_path: Path, language: str = "en") -> str:
        """Transcribe an audio file using whisper-cli."""
        if not WHISPER_CLI_PATH.exists():
            raise Exception(f"Whisper CLI not found at {WHISPER_CLI_PATH}")
        if not WHISPER_MODEL_PATH.exists():
            raise Exception(f"Whisper model not found at {WHISPER_MODEL_PATH}")

        # Command to transcribe to stdout
        cmd = [
            str(WHISPER_CLI_PATH),
            "-m", str(WHISPER_MODEL_PATH),
            "-f", str(audio_path),
            "-l", language,
            "-nt",       # No timestamps
            "-np",       # No prints (only the text)
        ]
        
        logger.info(f"Running Whisper transcription: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"Whisper Error: {error_msg}")
            raise Exception(f"Whisper failed with code {proc.returncode}")

        text = stdout.decode().strip()
        # Remove common Whisper artifacts like [Music], (silence), etc.
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text.strip()

class ModelManager:
    """Manages a persistent TTS daemon process with pipelined request handling.
    
    Architecture: A background _response_reader task continuously reads DONE/ERROR
    from the daemon's stdout and dispatches them to callers via a FIFO queue of
    asyncio Futures. The _write_lock is held only during the brief stdin write
    (microseconds), NOT for the entire synthesis duration. This allows multiple
    HTTP requests to have commands "in flight" simultaneously — the daemon's stdin
    buffer queues them and processes back-to-back with zero GPU idle gap.
    
    Before (old lock-based):
        HTTP-A: [acquire lock] [write cmd] [GPU works 2s] [read DONE] [release lock]
        HTTP-B:                                                        [acquire lock] [write cmd] [GPU works 2s] ...
                                                          ^^^^^^^^^^^^
                                                          GPU IDLE (lock contention + IPC)
    
    After (queue-based):
        HTTP-A: [write cmd] --------- [await future → resolved when DONE read] 
        HTTP-B:   [write cmd] ------- [await future → resolved when DONE read]
        Daemon: [process A] [process B]  ← back-to-back, zero idle gap
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.proc = None
        self._startup_lock = asyncio.Lock()  # Serializes daemon start/restart
        self._write_lock = asyncio.Lock()    # Serializes stdin writes (held briefly)
        self._pending: asyncio.Queue = asyncio.Queue()  # FIFO of Futures awaiting responses
        self._reader_task = None
        self._stderr_task = None

    async def start(self):
        async with self._startup_lock:
            if self.proc and self.proc.returncode is None:
                return

            # Cancel old tasks if restarting
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()

            info = get_model_capabilities(self.model_name)
            model_path = info.get("path", str(MODELS_BASE_DIR))
            model_file = info.get("model_name", self.model_name)
            
            cmd = [
                str(CLI_PATH),
                "-m", model_path,
                "--model-name", model_file,
                "--daemon"
            ]
            
            logger.info(f"Starting model daemon: {' '.join(cmd)}")
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
                
            logger.info(f"Model {self.model_name} is READY")
            
            # Start background tasks for reading responses and stderr
            self._reader_task = asyncio.create_task(self._response_reader())
            self._stderr_task = asyncio.create_task(self._log_stderr())

    async def _response_reader(self):
        """Background task: reads DONE/ERROR from daemon stdout and resolves pending Futures.
        
        This runs continuously and dispatches responses in FIFO order to match
        the order commands were written to stdin.
        """
        try:
            while self.proc and self.proc.returncode is None:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                
                resp = line.decode().strip()
                
                try:
                    future = self._pending.get_nowait()
                except asyncio.QueueEmpty:
                    logger.warning(f"[{self.model_name}] Unexpected daemon output (no waiter): {resp}")
                    continue
                
                if future.done():
                    continue
                
                if resp.startswith("DONE|"):
                    future.set_result((True, resp.split("|", 1)[1]))
                elif resp.startswith("ERROR|"):
                    future.set_result((False, resp.split("|", 1)[1]))
                else:
                    future.set_result((False, f"Unexpected response: {resp}"))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.model_name}] Response reader error: {e}")
        finally:
            # Daemon died or reader cancelled — fail all pending futures
            self._fail_all_pending("Model process terminated unexpectedly")

    def _fail_all_pending(self, error_msg: str):
        """Resolve all pending Futures with an error (daemon died)."""
        while not self._pending.empty():
            try:
                future = self._pending.get_nowait()
                if not future.done():
                    future.set_result((False, error_msg))
            except asyncio.QueueEmpty:
                break

    async def _log_stderr(self):
        try:
            while self.proc and self.proc.returncode is None:
                line = await self.proc.stderr.readline()
                if not line: break
                logger.info(f"[{self.model_name}] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass

    async def _ensure_running(self):
        """Ensure daemon is running, restart if dead."""
        if not self.proc or self.proc.returncode is not None:
            await self.start()

    async def synthesize(self, text, output, speaker="", reference="", instruct="", embedding=""):
        """Submit a single synthesis command and await the result."""
        future = await self.submit_synthesis(text, output, speaker, reference, instruct, embedding)
        result = await future
        
        # If daemon died, try restart + one retry
        if not result[0] and (not self.proc or self.proc.returncode is not None):
            logger.warning(f"Daemon died during synthesis, restarting for retry...")
            await self.start()
            return await self._synthesize_no_retry(text, output, speaker, reference, instruct, embedding)
        
        return result

    async def submit_synthesis(self, text, output, speaker="", reference="", instruct="", embedding=""):
        """Submit a synthesis command and return the Future immediately.
        Allows the caller to await results as needed (e.g. for streaming).
        """
        await self._ensure_running()
        
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        async with self._write_lock:
            if not self.proc or self.proc.returncode is not None:
                await self.start()
            
            cmd_line = f"{text}|{output}|{speaker}|{reference}|{instruct}|{embedding}\n"
            logger.info(f"[{self.model_name}] → daemon: text=[{len(text)}c]{text[:50]}… ref={'Y' if reference else 'N'} inst=[{len(instruct)}c] emb={'Y' if embedding else 'N'}")
            self.proc.stdin.write(cmd_line.encode())
            await self.proc.stdin.drain()
            await self._pending.put(future)
            
        return future

    async def _synthesize_no_retry(self, text, output, speaker="", reference="", instruct="", embedding=""):
        """Single attempt, no retry (used after restart to avoid infinite loops)."""
        await self._ensure_running()
        
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        async with self._write_lock:
            if not self.proc or self.proc.returncode is not None:
                return False, "Model process not running"
            cmd_line = f"{text}|{output}|{speaker}|{reference}|{instruct}|{embedding}\n"
            self.proc.stdin.write(cmd_line.encode())
            await self.proc.stdin.drain()
            await self._pending.put(future)
        
        return await future

    async def synthesize_batch(self, requests: list):
        """Pipeline multiple commands to stdin at once. Works both within a single
        HTTP request (internal chunking) and across concurrent HTTP requests.
        
        All commands are written under a single _write_lock acquisition, ensuring
        they are contiguous in the daemon's stdin buffer.
        """
        if not requests:
            return []
        
        if len(requests) == 1:
            r = requests[0]
            return [await self.synthesize(
                r['text'], r['output'], r.get('speaker', ''),
                r.get('reference', ''), r.get('instruct', ''),
                r.get('embedding', '')
            )]

        await self._ensure_running()
        
        loop = asyncio.get_event_loop()
        futures = []
        
        async with self._write_lock:
            if not self.proc or self.proc.returncode is not None:
                await self.start()
            
            for r in requests:
                future = loop.create_future()
                cmd_line = "{text}|{output}|{speaker}|{reference}|{instruct}|{embedding}\n".format(
                    text=r['text'], output=r['output'],
                    speaker=r.get('speaker', ''), reference=r.get('reference', ''),
                    instruct=r.get('instruct', ''), embedding=r.get('embedding', '')
                )
                self.proc.stdin.write(cmd_line.encode())
                await self._pending.put(future)
                futures.append(future)
            await self.proc.stdin.drain()
        
        # Await all results — write_lock released, other requests can pipeline
        return [await f for f in futures]

async def get_manager(model_name: str) -> ModelManager:
    async with manager_lock:
        if model_name not in managers:
            managers[model_name] = ModelManager(model_name)
        manager = managers[model_name]
    await manager.start()
    return manager

def split_text(text: str, lang: str = "en") -> List[str]:
    return normalizer.split_sentences(text, lang)

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not ENABLE_TTS:
        raise HTTPException(status_code=400, detail="TTS engine is disabled on this server instance.")
    
    model_name = req.model or DEFAULT_MODEL
    voice_name = req.voice
    instruction = req.extra_body.get("instruction") if req.extra_body else None

    # 1. Professional Normalization
    text, detected_lang = normalizer.process(req.input)
    
    info = get_model_capabilities(model_name)
    current_presets = load_presets()
    if voice_name in current_presets:
        preset = current_presets[voice_name]
        voice_name = preset.get("voice", voice_name)
        if not instruction:
            instruction = preset.get("instruction")

    # 2. Sentence-based splitting
    text_chunks = normalizer.split_sentences(text, detected_lang)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="Empty input text")

    logger.info(f"Input: {len(req.input)} chars → Normalized: {len(text)} chars → {len(text_chunks)} chunks (lang={detected_lang})")
    for i, chunk in enumerate(text_chunks):
        logger.debug(f"  Chunk {i}: [{len(chunk)} chars] {chunk[:80]}...")

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
            embed_cache_path = EMBED_CACHE_DIR / f"{voice_name}_{model_name}_{wav_hash}.json"
            transcript_cache_path = TRANSCRIPT_CACHE_DIR / f"{voice_name}_{wav_hash}.txt"
            processed_wav_path = EMBED_CACHE_DIR / f"processed_{wav_hash}.wav"
            
            # Normalize audio
            if not processed_wav_path.exists():
                try:
                    await normalize_audio(reference_wav, processed_wav_path)
                except:
                    processed_wav_path = reference_wav
            
            # Transcription (cached)
            auto_transcript = ""
            if transcript_cache_path.exists():
                try:
                    auto_transcript = transcript_cache_path.read_text().strip()
                except: pass
            
            if not auto_transcript:
                try:
                    auto_transcript = await WhisperManager.transcribe(processed_wav_path)
                    if auto_transcript:
                        transcript_cache_path.write_text(auto_transcript)
                except: pass
            
            if auto_transcript:
                auto_transcript = auto_transcript.replace("\n", " ").replace("\r", " ").replace("|", " ")
                # Trim to complete sentences — truncated audio produces fragments
                # that start/end mid-sentence, breaking voice cloning alignment
                try:
                    import pysbd
                    seg = pysbd.Segmenter(language="en", clean=False)
                    sentences = seg.segment(auto_transcript)
                    # Drop first sentence if it starts lowercase (mid-sentence fragment)
                    if sentences and sentences[0] and sentences[0][0].islower():
                        logger.info(f"Trimming leading fragment: {sentences[0][:40]}...")
                        sentences = sentences[1:]
                    # Drop last sentence if it doesn't end with sentence-ending punctuation
                    if sentences and sentences[-1] and sentences[-1].strip()[-1] not in '.!?':
                        logger.info(f"Trimming trailing fragment: ...{sentences[-1][-40:]}")
                        sentences = sentences[:-1]
                    if sentences:
                        auto_transcript = " ".join(s.strip() for s in sentences)
                    else:
                        logger.warning("All transcript sentences were fragments, using raw transcript")
                except Exception as e:
                    logger.warning(f"Sentence trimming failed: {e}")
                
                if instruction:
                    instruction = f"{auto_transcript}. {instruction}"
                else:
                    instruction = auto_transcript

            if embed_cache_path.exists():
                embedding = str(embed_cache_path)
            else:
                reference = str(processed_wav_path)
                embedding = str(embed_cache_path)
        else:
            if info and info.get("speakers"):
                speaker = info.get("speakers")[0]
            else:
                speaker = "vivian"

    async def audio_generator():
        needs_embedding_first = bool(reference)
        temp_files = []
        try:
            # First chunk
            first_text = text_chunks[0]
            first_wav = Path(f"/tmp/chunk_{request_id}_0.wav")
            temp_files.append(first_wav)
            
            # 1. Synthesize chunk 0
            logger.info(f"Synthesizing first chunk: {first_text[:30]}...")
            first_future = await manager.submit_synthesis(
                first_text, str(first_wav), speaker, reference, instruction or "", embedding
            )
            
            if needs_embedding_first:
                # Voice clone: chunk 0 computes the embedding file.
                # We MUST await it before submitting chunks 1-N, otherwise
                # they try to load a non-existent embedding and fail.
                logger.info("New voice clone — awaiting first chunk to compute embedding...")
                success, result = await first_future
                
                if success and first_wav.exists():
                    mp3_data = await convert_wav_to_mp3(first_wav, apply_fades=True)
                    yield mp3_data
                else:
                    logger.error(f"First chunk (embedding) failed: {result}")
                    return
                
                # Now pipeline the remaining chunks (embedding file exists)
                background_futures = []
                for i in range(1, len(text_chunks)):
                    chunk_wav = Path(f"/tmp/chunk_{request_id}_{i}.wav")
                    temp_files.append(chunk_wav)
                    future = await manager.submit_synthesis(
                        text_chunks[i], str(chunk_wav), speaker, "", instruction or "", embedding
                    )
                    background_futures.append(future)
                
                for i, future in enumerate(background_futures):
                    chunk_idx = i + 1
                    chunk_wav = Path(f"/tmp/chunk_{request_id}_{chunk_idx}.wav")
                    success, result = await future
                    if success and chunk_wav.exists():
                        mp3_data = await convert_wav_to_mp3(chunk_wav, apply_fades=True)
                        yield mp3_data
                    else:
                        logger.error(f"Chunk {chunk_idx} failed: {result}")
            else:
                # Precomputed embedding or internal speaker: safe to pipeline everything
                background_futures = []
                for i in range(1, len(text_chunks)):
                    chunk_wav = Path(f"/tmp/chunk_{request_id}_{i}.wav")
                    temp_files.append(chunk_wav)
                    future = await manager.submit_synthesis(
                        text_chunks[i], str(chunk_wav), speaker, "", instruction or "", embedding
                    )
                    background_futures.append(future)
                
                # Await and stream chunk 0
                success, result = await first_future
                if success and first_wav.exists():
                    mp3_data = await convert_wav_to_mp3(first_wav, apply_fades=True)
                    yield mp3_data
                else:
                    logger.error(f"First chunk failed: {result}")
                
                # Stream the rest as they finish
                for i, future in enumerate(background_futures):
                    chunk_idx = i + 1
                    chunk_wav = Path(f"/tmp/chunk_{request_id}_{chunk_idx}.wav")
                    success, result = await future
                    if success and chunk_wav.exists():
                        mp3_data = await convert_wav_to_mp3(chunk_wav, apply_fades=True)
                        yield mp3_data
                    else:
                        logger.error(f"Chunk {chunk_idx} failed: {result}")
                    
        finally:
            # Cleanup all temp files
            for f in temp_files:
                if f.exists():
                    try: os.remove(f)
                    except: pass

    return StreamingResponse(audio_generator(), media_type="audio/mpeg")

async def convert_wav_to_mp3(wav_path: Path, apply_fades: bool = True) -> bytes:
    """Convert WAV to MP3 and apply 5ms fades to prevent clicks between chunks."""
    if not wav_path.exists():
        return b""
        
    duration = 0.0
    if apply_fades:
        try:
            # Get duration using ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)
            ]
            proc = await asyncio.create_subprocess_exec(
                *probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await proc.communicate()
            duration = float(stdout.decode().strip())
        except:
            apply_fades = False

    cmd = ["ffmpeg", "-i", str(wav_path)]
    if apply_fades and duration > 0.01:
        # Apply 5ms fade-in and 5ms fade-out
        fade_out_start = max(0, duration - 0.005)
        cmd.extend(["-af", f"afade=t=in:st=0:d=0.005,afade=t=out:st={fade_out_start}:d=0.005"])

    cmd.extend(["-codec:a", "libmp3lame", "-qscale:a", "2", "-f", "mp3", "pipe:1"])
    
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    stdout, _ = await proc.communicate()
    return stdout

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
