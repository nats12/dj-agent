"""AI-powered musical reasoning — vibe analysis, transition advice, nuance tagging.

Uses a multimodal LLM to provide high-level musical understanding that
numerical features alone cannot capture.

Backend priority (auto mode):
1. Audio Flamingo 3 (local GPU, expert audio reasoning) — if CUDA + transformers
2. Ollama (local, free) — only if multimodal model available
3. Gemini SDK (cloud, cheap, 1M context) — if GOOGLE_API_KEY set

Note: The Gemini CLI is NOT used as a transport. It is an autonomous agent
that triggers recursive loops and stdout pollution when invoked as a subprocess.
Only the google-genai SDK is used for Gemini queries.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import re

import numpy as np

from .config import get_config


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Check if Ollama is running with a multimodal model.

    Just having Ollama running is not enough — the configured model must
    support audio/image input. Text-only models will 500 on audio.
    """
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False

        # Check if any model in the list is known-multimodal
        models = r.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        config = get_config().reasoning
        target = os.environ.get("OLLAMA_MODEL", config.ollama_model)

        # Verify the target model exists locally
        if not any(target in name for name in model_names):
            return False

        # Known multimodal model families that accept audio/images
        _MULTIMODAL_PATTERNS = (
            "llava", "bakllava", "moondream", "qwen2-audio", "qwen2.5-omni",
            "gemma3", "llama3.2-vision", "minicpm-v",
        )
        # qwen3.5 is text-only — do NOT send audio to it
        if any(pat in target.lower() for pat in _MULTIMODAL_PATTERNS):
            return True

        # If unsure, don't risk a 500 — prefer Flamingo or Gemini
        return False

    except Exception:
        return False


_flamingo_verified: bool | None = None  # cache: True/False/None=untested


def _flamingo_available() -> bool:
    """Check if Audio Flamingo dependencies exist locally (no network calls).

    Only checks imports and CUDA — does NOT hit HuggingFace.
    Actual model loading happens lazily on first query, not during detection.
    """
    global _flamingo_verified
    if _flamingo_verified is not None:
        return _flamingo_verified

    try:
        import torch
        if not torch.cuda.is_available():
            _flamingo_verified = False
            return False
        # Check transformers is importable (no network call)
        import transformers  # noqa: F401
        _flamingo_verified = True
        return True
    except ImportError:
        _flamingo_verified = False
        return False


def _gemini_available() -> bool:
    """Check if Gemini SDK can be used (API key or .env file).

    Note: Gemini CLI OAuth is NOT checked here — the CLI is an autonomous
    agent that causes recursive loops when called as a subprocess.
    Only the SDK with an API key is a reliable audio transport.
    """
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return True
    # Check .env files for API key
    for env_path in [Path.cwd() / ".env", Path.home() / ".env"]:
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    if line.startswith("GOOGLE_API_KEY=") and line.split("=", 1)[1].strip():
                        return True
            except Exception:
                pass
    return False


_demoted_backends: set[str] = set()  # backends that failed at runtime this session


def demote_backend(name: str, reason: str) -> None:
    """Mark a backend as failed for this session. It won't be auto-selected again."""
    _demoted_backends.add(name)
    import logging
    logging.getLogger(__name__).warning("Backend '%s' demoted: %s", name, reason)


def get_backend() -> str:
    """Return the best available backend, skipping any that failed this session."""
    if "flamingo" not in _demoted_backends and _flamingo_available():
        import torch
        if torch.cuda.is_available():
            return "flamingo"
    if "ollama" not in _demoted_backends and _ollama_available():
        return "ollama"
    if "gemini" not in _demoted_backends and _gemini_available():
        return "gemini"
    return "none"


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM responses
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response that may contain chatter.

    Handles: markdown code blocks, preamble text ("Sure, here is..."),
    trailing explanations, and mixed text/JSON responses.
    Always returns a dict (wraps arrays/scalars if needed).
    """
    raw = raw.strip()

    def _ensure_dict(obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        return {"data": obj}  # wrap non-dict JSON (arrays, scalars)

    # 1. Try direct parse first (clean response)
    try:
        return _ensure_dict(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code blocks
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if match:
            try:
                return _ensure_dict(json.loads(match.group(1).strip()))
            except (json.JSONDecodeError, ValueError):
                pass

    # 3. Find first { ... } block in the response (supports nested braces)
    match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, re.DOTALL)
    if match:
        try:
            return _ensure_dict(json.loads(match.group(0)))
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Give up — return raw as fallback
    return {"raw_response": raw}


# ---------------------------------------------------------------------------
# Startup cleanup — remove orphaned temp files from crashed sessions
# ---------------------------------------------------------------------------

_TEMP_PREFIX = f"dj_reason_{os.getpid()}_"
_TEMP_MIX_PREFIX = f"dj_mix_{os.getpid()}_"


def cleanup_temp_snippets() -> int:
    """Remove orphaned temp files from dead processes.

    Uses PID-based ownership: only deletes files whose embedded PID
    no longer corresponds to a running process.
    """
    import glob

    count = 0
    for pattern in ["/tmp/dj_reason_*", "/tmp/dj_mix_*"]:
        for f in glob.glob(pattern):
            try:
                # Extract PID from filename: dj_reason_{PID}_xxxxx.wav
                parts = Path(f).stem.split("_")
                if len(parts) >= 3:
                    try:
                        pid = int(parts[2])
                    except ValueError:
                        pid = None  # legacy format — no PID
                    if pid is not None:
                        try:
                            os.kill(pid, 0)  # check if process exists
                            continue  # process alive — don't delete
                        except ProcessLookupError:
                            pass  # process dead — safe to delete
                        except (PermissionError, OSError):
                            continue  # can't check — leave alone
                # Old format or can't parse PID — only delete if older than 1 hour
                import time
                age = time.time() - Path(f).stat().st_mtime
                if age > 3600:
                    Path(f).unlink(missing_ok=True)
                    count += 1
            except OSError:
                pass
    return count


import atexit
atexit.register(cleanup_temp_snippets)


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def _get_duration(path: Path) -> float:
    """Get track duration in seconds without loading the full file.

    Uses ffprobe if available, falls back to mutagen, then librosa.
    Returns 180.0 (3 min) as last resort if all methods fail.
    """
    # Try ffprobe (fastest, no Python decode)
    import subprocess as _sp
    try:
        probe = _sp.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return float(probe.stdout.strip())
    except Exception:
        pass

    # Try mutagen (fast, no audio decode)
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(str(path))
        if mf and mf.info and mf.info.length:
            return float(mf.info.length)
    except Exception:
        pass

    # Last resort: librosa (loads audio — slow but always works)
    try:
        import librosa
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 180.0  # conservative 3-minute default

def _extract_snippet(
    path: str | Path,
    duration_sec: float | None = None,
    offset_pct: float = 0.25,
    sr: int = 16000,
) -> Path:
    """Extract a representative snippet from a track.

    When ``offset_pct`` is explicitly set (e.g. 0.75 for end-of-track),
    extracts a single clip at that position.  Otherwise samples from TWO
    points (25% and 50%) to avoid the "snippet bias" problem.
    Downsamples to 16kHz mono WAV.
    """
    import soundfile as sf
    import librosa

    if duration_sec is None:
        duration_sec = get_config().reasoning.snippet_duration_sec

    path = Path(path)
    # Get actual duration without loading full file
    total = _get_duration(path)

    # If caller specified an explicit offset (e.g. suggest_transition uses
    # 0.75 for end-of-track and 0.0 for start), honour it as a single clip.
    # Otherwise use dual-point sampling for vibe analysis.
    if offset_pct not in (0.25, None):
        # Explicit offset — single clip
        offsets = [total * offset_pct]
        half_dur = duration_sec
    else:
        # Default: dual-point sampling
        offsets = [total * 0.25, total * 0.50]
        half_dur = duration_sec / 2.0

    segments = []
    for off in offsets:
        y, _ = librosa.load(str(path), sr=sr, mono=True,
                             offset=off, duration=half_dur)
        segments.append(y)

    snippet = np.concatenate(segments) if len(segments) > 1 else segments[0]

    tmp = Path(tempfile.NamedTemporaryFile(suffix=".wav", prefix=_TEMP_PREFIX,
                                            delete=False).name)
    sf.write(str(tmp), snippet, sr)
    return tmp


def _audio_to_base64(path: Path) -> str:
    """Encode audio file as base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Audio Flamingo backend (Local transformers)
# ---------------------------------------------------------------------------

_FLAMINGO_MODEL = None
_FLAMINGO_PROCESSOR = None
_FLAMINGO_LOCK = threading.Lock()


def _load_flamingo() -> tuple[Any, Any]:
    """Lazy-load Audio Flamingo model and processor."""
    global _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR
    if _FLAMINGO_MODEL is not None:
        return _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR

    with _FLAMINGO_LOCK:
        if _FLAMINGO_MODEL is not None:
            return _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR

        import torch
        from transformers import (
            AutoProcessor,
            AudioFlamingo3ForConditionalGeneration,
            BitsAndBytesConfig,
        )

        config = get_config().reasoning
        model_id = config.model_id

        # Quantization settings
        bnb_config = None
        if config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        trust = config.trust_remote_code
        _FLAMINGO_PROCESSOR = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust
        )
        _FLAMINGO_MODEL = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust,
            torch_dtype=torch.float16 if bnb_config else torch.float32,
        )
        return _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR


def _unload_flamingo() -> None:
    """Unload Flamingo from GPU to free VRAM."""
    global _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR
    _FLAMINGO_MODEL = None
    _FLAMINGO_PROCESSOR = None


# Register with GPU manager
try:
    from .gpu import gpu_manager as _gpu
    _gpu.register_unloader("flamingo", _unload_flamingo)
except Exception:
    pass


def _flamingo_query(audio_path: Path, prompt: str) -> str:
    """Send audio + prompt to local Audio Flamingo."""
    from .gpu import gpu_manager
    gpu_manager._ensure_owner("flamingo")
    model, processor = _load_flamingo()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio_path)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    
    # Cast float32 inputs to model's dtype (e.g. float16) to avoid mismatch
    import torch
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            inputs[k] = v.to(model.dtype)
        inputs[k] = inputs[k].to(model.device)

    config = get_config().reasoning
    outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)

    # Decode only the generated part
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0].strip()


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "qwen3.5:27b"  # override with OLLAMA_MODEL env var


def _ollama_query(audio_path: Path, prompt: str) -> str:
    """Send audio + prompt to local Ollama."""
    import requests

    config = get_config().reasoning
    model = os.environ.get("OLLAMA_MODEL", config.ollama_model)
    audio_b64 = _audio_to_base64(audio_path)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [audio_b64],  # Ollama multimodal uses "images" for all media
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

# Model tiers for different use cases
GEMINI_MODELS = {
    "lite": "gemini-3.1-flash-lite-preview",   # $0.25/M — batch tagging
    "flash": "gemini-3-flash-preview",          # $0.50/M — default analysis
    "pro": "gemini-3.1-pro-preview",            # $2.00/M — deep reasoning
}


def setup_gemini(api_key: str) -> bool:
    """Store Gemini API key for this session (only if verification succeeds).

    Users can get a free key at https://aistudio.google.com/apikey
    """
    # Verify BEFORE storing — don't poison env with bad keys
    try:
        from google import genai  # type: ignore[import-untyped]
        client = genai.Client(api_key=api_key)
        client.models.list()
        os.environ["GOOGLE_API_KEY"] = api_key
        return True
    except Exception:
        return False


def _gemini_query(
    audio_path: Path,
    prompt: str,
    model_tier: str = "flash",
) -> str:
    """Send audio + prompt to Gemini via the SDK (direct API, no subprocess).

    The Gemini CLI is NOT used — it is an autonomous agent that can trigger
    recursive loops, stdout pollution, and unpredictable tool execution.
    The SDK sends audio bytes directly via the API, which is reliable and
    deterministic.

    Parameters
    ----------
    model_tier : "lite", "flash", "pro"
    """
    return _gemini_sdk_query(audio_path, prompt, model_tier)


def _gemini_sdk_query(
    audio_path: Path,
    prompt: str,
    model_tier: str,
) -> str:
    """Query via google-genai SDK (API key or OAuth/ADC auto-discovery)."""
    try:
        from google import genai  # type: ignore[import-untyped]
        from google.genai import types
    except ImportError:
        raise ImportError(
            "pip install google-genai\n"
            "Then authenticate via one of:\n"
            "  1. Set GOOGLE_API_KEY (get free key at https://aistudio.google.com/apikey)\n"
            "  2. Run 'gemini auth' for Google account OAuth login\n"
            "  3. Run 'gcloud auth application-default login' for ADC"
        )

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    # Try to load from .env file if not in environment
    if not api_key:
        for env_path in [
            Path.cwd() / ".env",
            Path.home() / ".env",
            Path.home() / ".dj-agent" / ".env",
        ]:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().split("#")[0].strip()
                        break
            if api_key:
                break

    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        # Check if we can use ADC
        try:
            client = genai.Client()
        except Exception:
            raise RuntimeError(
                "Gemini API key required for audio analysis fallback.\n"
                "Set GOOGLE_API_KEY in environment or .env file.\n"
                "Get a free key at https://aistudio.google.com/apikey"
            )

    model = GEMINI_MODELS.get(model_tier, GEMINI_MODELS["flash"])

    # Guard: reject files larger than 20MB (Gemini inline limit)
    # All snippets from _extract_snippet are ~1MB; this catches accidental raw file paths
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 20:
        raise ValueError(
            f"Audio file too large for inline Gemini query ({file_size_mb:.0f}MB). "
            "Use _extract_snippet() first, or pass a shorter clip."
        )

    mime = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"
    audio_bytes = audio_path.read_bytes()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                ]
            )
        ],
    )
    return response.text


# ---------------------------------------------------------------------------
# Unified query
# ---------------------------------------------------------------------------

def _query(
    audio_path: Path,
    prompt: str,
    backend: str = "auto",
    model_tier: str = "flash",
) -> str:
    """Send audio + prompt to the best available backend.

    Parameters
    ----------
    backend : "auto", "flamingo", "ollama", "gemini", "gemini-lite", "gemini-pro"
    model_tier : "lite", "flash", "pro" (Gemini only)
    """
    if backend == "auto":
        config_backend = get_config().reasoning.backend
        backend = config_backend if config_backend != "auto" else get_backend()

    # Parse backend shortcuts
    if backend.startswith("gemini-"):
        model_tier = backend.split("-", 1)[1]  # "gemini-pro" → "pro"
        backend = "gemini"

    # Universal size guard — applies to ALL backends (not just Gemini)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 20:
        raise ValueError(
            f"Audio file too large ({file_size_mb:.0f}MB). "
            "Use _extract_snippet() first, or pass a shorter clip."
        )

    # Try the selected backend; on failure, demote and retry with next best
    try:
        if backend == "flamingo":
            return _flamingo_query(audio_path, prompt)
        elif backend == "ollama":
            return _ollama_query(audio_path, prompt)
        elif backend == "gemini":
            return _gemini_query(audio_path, prompt, model_tier=model_tier)
    except Exception as e:
        demote_backend(backend, str(e)[:100])
        # Retry with next best backend
        next_backend = get_backend()
        if next_backend != "none" and next_backend != backend:
            return _query(audio_path, prompt, backend=next_backend, model_tier=model_tier)

    raise RuntimeError(
            "No reasoning backend available. "
            "Install transformers for Flamingo, start Ollama, or set GOOGLE_API_KEY."
        )


# ---------------------------------------------------------------------------
# Public API — Musical reasoning functions
# ---------------------------------------------------------------------------

DJ_PERSONA = (
    "You are a world-class DJ and music analyst with 20 years of experience "
    "in electronic music. Be concise and specific. "
)


def analyze_vibe(path: str | Path, backend: str = "auto") -> str:
    """Describe the 'vibe' of a track — texture, setting, energy.

    Returns a short text description like "Dark, hypnotic warehouse techno.
    Peak-time energy with rolling bass and chopped vocal stabs."
    """
    snippet = _extract_snippet(path)
    try:
        return _query(snippet, DJ_PERSONA + (
            "Analyze this audio clip. Describe the 'vibe' in 2-3 sentences. "
            "Focus on: texture, aggression level, and the ideal dancefloor "
            "setting (e.g., 'warm-up', 'peak-time', 'after-hours')."
        ), backend)
    finally:
        snippet.unlink(missing_ok=True)


def get_energy_arc(path: str | Path, backend: str = "auto") -> str:
    """Describe how energy changes through the clip.

    Returns labels like "Rising", "Driving", "Floating", "Dropping".
    """
    snippet = _extract_snippet(path)
    try:
        return _query(snippet, DJ_PERSONA + (
            "Listen to this clip. How does the energy change? "
            "Classify as one of: 'Driving' (constant high), 'Rising' (building), "
            "'Floating' (ambient/sustained), 'Dropping' (energy decreasing). "
            "Explain in one sentence why."
        ), backend)
    finally:
        snippet.unlink(missing_ok=True)


def suggest_transition(
    path_a: str | Path,
    path_b: str | Path,
    backend: str = "auto",
) -> str:
    """Suggest the best transition technique between two tracks."""
    snip_a = _extract_snippet(path_a, offset_pct=0.75)  # end of track A
    snip_b = _extract_snippet(path_b, offset_pct=0.0)   # start of track B
    combined_path: Path | None = None
    try:
        import soundfile as sf
        a_data, sr = sf.read(str(snip_a))
        b_data, _ = sf.read(str(snip_b))
        combined = np.concatenate([a_data, np.zeros(sr), b_data])  # 1s gap
        # Use NamedTemporaryFile (not mktemp) to avoid race condition
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix=_TEMP_MIX_PREFIX, delete=False)
        combined_path = Path(tmp.name)
        tmp.close()
        sf.write(str(combined_path), combined, sr)

        return _query(combined_path, DJ_PERSONA + (
            "This clip contains the end of Track A, a brief silence, then the "
            "start of Track B. Suggest the best DJ transition technique "
            "(e.g., 'smooth EQ blend over 16 bars', 'hard cut on the 1', "
            "'filter sweep into drop'). Explain why in 2 sentences."
        ), backend)
    finally:
        # Clean up ALL temp files — even on exception
        snip_a.unlink(missing_ok=True)
        snip_b.unlink(missing_ok=True)
        if combined_path:
            combined_path.unlink(missing_ok=True)


def classify_nuance(path: str | Path, backend: str = "auto") -> dict[str, str]:
    """Extract nuanced musical tags that audio features can't capture.

    Returns tags like bassline_type, vocal_style, rhythm_feel, mood_detail.
    """
    snippet = _extract_snippet(path)
    try:
        raw = _query(snippet, DJ_PERSONA + (
            "Analyze this audio. Return ONLY a JSON object with these keys:\n"
            '- "bassline": one of "rolling", "off-beat", "syncopated", "minimal", "heavy"\n'
            '- "vocals": one of "none", "chopped", "soulful", "atmospheric", "MC", "rap"\n'
            '- "rhythm": one of "four-on-the-floor", "broken-beat", "shuffle", "polyrhythmic"\n'
            '- "mood": one of "dark", "euphoric", "melancholic", "aggressive", "groovy", "hypnotic"\n'
            '- "setting": one of "warm-up", "peak-time", "after-hours", "festival", "underground"\n'
            "Respond with ONLY the JSON, no explanation."
        ), backend)

        # Robust JSON extraction — handles markdown blocks, preamble chatter,
        # and mixed text/JSON responses from different backends
        return _extract_json(raw)
    finally:
        snippet.unlink(missing_ok=True)
