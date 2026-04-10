"""Stem separation & export — vocals, drums, bass, other, or *anything*.

Model selection (2026):
- **BS-RoFormer (12.97 SDR)** — top single-file vocal model on MVSep/MUSDB18
  leaderboards. 2-stem (vocals + instrumental) in one pass, which beats
  summing 4 Demucs stems (leakage compounds on sum).
- **Mel-Band RoFormer (11.44 SDR)** — cleaner on exposed pop vocals; a bit
  less robust on buried/electronic vocals. Kept as an alternative.
- **HTDemucs** — general 4/6-stem separator. Good for drums/bass but its
  vocal SDR (8.4) is now significantly behind RoFormer.
- **SAM Audio** (Meta, Dec 2025) — text-prompted foundation model. Isolate
  *any* sound by description ("synth pad", "hi-hats", "crowd noise").
  One stem per inference. Best for non-standard separations.

Quality presets:
- ``fast``     : HTDemucs 4-stem (~10 s/track on RTX 3090)
- ``balanced`` : BS-RoFormer 2-stem (~25 s/track) — DEFAULT
- ``best``     : BS-RoFormer + Mel-Band ensemble (~50 s/track, ~+0.3 dB)
- ``sam``      : SAM Audio text-prompted (~30 s/stem, flexible)

audio-separator auto-downloads model weights on first use.
SAM Audio requires ``pip install sam-audio`` + ``huggingface-cli login``.
"""

from __future__ import annotations
import logging

log = logging.getLogger(__name__)

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

def _has_audio_separator() -> bool:
    try:
        from audio_separator.separator import Separator  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def _has_demucs() -> bool:
    try:
        import demucs.api  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def _has_sam_audio() -> bool:
    try:
        _ensure_torchcodec_importable()
        from sam_audio import SAMAudio  # type: ignore[import-untyped]
        return True
    except (ImportError, RuntimeError):
        return False


def _ensure_torchcodec_importable() -> None:
    """Stub out torchcodec if its native libs are missing.

    SAM Audio eagerly imports torchcodec (a video decoder) at module level,
    even for audio-only text-prompted separation. If the CUDA NPP libs
    aren't installed, torchcodec raises RuntimeError on import.  We mock
    it so SAM Audio can load — video-prompted separation won't work, but
    we only use text prompts.
    """
    import sys
    if "torchcodec" in sys.modules:
        return
    try:
        import torchcodec  # type: ignore[import-untyped]  # noqa: F401
        # Verify it actually loaded (not just found the spec)
        torchcodec.decoders  # noqa: B018
    except (RuntimeError, OSError, AttributeError, ValueError):
        import types
        import importlib

        tc = types.ModuleType("torchcodec")
        tc.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
        tc.__version__ = "0.0.0"
        tc.decoders = types.ModuleType("torchcodec.decoders")  # type: ignore[attr-defined]
        tc.decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", None)  # type: ignore[attr-defined]
        tc.decoders.AudioDecoder = None  # type: ignore[attr-defined]
        tc.decoders.VideoDecoder = None  # type: ignore[attr-defined]
        sys.modules["torchcodec"] = tc
        sys.modules["torchcodec.decoders"] = tc.decoders  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Roformer separation (preferred for vocals)
# ---------------------------------------------------------------------------

# Known good models — auto-downloaded by audio-separator on first use.
#
# BS-RoFormer (ep 317, SDR 12.9755) is the top single-file vocal/instrumental
# model as of 2026 — a 2-stem model whose single inference yields BOTH vocals
# and instrumental. That's why it's our default for both create_acapella and
# create_instrumental.
ROFORMER_BS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_MEL_BAND_MODEL = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"

# Backwards-compat aliases for callers that imported the old names.
ROFORMER_VOCAL_MODEL = ROFORMER_BS_MODEL
ROFORMER_INSTRUM_MODEL = ROFORMER_BS_MODEL


_roformer_cache: dict[str, Any] = {}
_roformer_lock = __import__("threading").Lock()


def _unload_roformer() -> None:
    """Unload all Roformer models from memory."""
    _roformer_cache.clear()


def _unload_demucs() -> None:
    """Unload all Demucs models from memory."""
    _demucs_cache.clear()


# Register with GPU manager
try:
    from .gpu import gpu_manager as _gpu_stems
    _gpu_stems.register_unloader("roformer", _unload_roformer)
    _gpu_stems.register_unloader("demucs", _unload_demucs)
except Exception:
    pass


def _get_roformer_separator(model_filename: str):
    """Cached Roformer separator (model loaded once per model name)."""
    if model_filename in _roformer_cache:
        return _roformer_cache[model_filename]
    with _roformer_lock:
        if model_filename in _roformer_cache:
            return _roformer_cache[model_filename]
        from audio_separator.separator import Separator  # type: ignore[import-untyped]
        sep = Separator()
        sep.load_model(model_filename=model_filename)
        _roformer_cache[model_filename] = sep
        return sep


def _separate_roformer(
    path: Path,
    model_filename: str = ROFORMER_BS_MODEL,
) -> dict[str, np.ndarray]:
    """Separate using audio-separator with a Roformer model."""
    try:
        from .gpu import gpu_manager
        gpu_manager._ensure_owner("roformer")
    except Exception:
        pass
    separator = _get_roformer_separator(model_filename)
    output_files = separator.separate(str(path))

    # audio-separator returns list of output file paths — read then clean up
    stems: dict[str, np.ndarray] = {}
    try:
        for fpath in output_files:
            p = Path(fpath)
            name = _parse_stem_name(p.stem)
            data, sr = sf.read(str(p))
            if data.ndim == 1:
                data = data[:, np.newaxis]
            stems[name] = data
    finally:
        # Clean up temp files written by audio-separator
        for fpath in output_files:
            try:
                Path(fpath).unlink(missing_ok=True)
            except OSError:
                pass

    return stems


def _parse_stem_name(filename_stem: str) -> str:
    """Extract a normalised stem name from an audio-separator output filename.

    audio-separator writes files like ``track_(Vocals)`` or ``track_(Instrumental)``
    so we look for the parenthesised suffix first. Falls back to substring
    matching if the convention changes.
    """
    import re
    match = re.search(r"\(([^)]+)\)\s*$", filename_stem)
    label = (match.group(1) if match else filename_stem).lower()
    if "vocal" in label or "voice" in label:
        return "vocals"
    if "instrument" in label or "no_vocal" in label or "no vocal" in label:
        return "instrumental"
    if "drum" in label:
        return "drums"
    if "bass" in label:
        return "bass"
    return label


# ---------------------------------------------------------------------------
# SAM Audio separation (text-prompted, any sound)
# ---------------------------------------------------------------------------

# DJ-friendly prompt aliases — map short names to descriptions that work
# well with SAM Audio's text encoder.
SAM_PROMPT_MAP: dict[str, str] = {
    "vocals": "singing voice",
    "drums": "drum kit playing",
    "bass": "bass guitar",
    "guitar": "electric guitar",
    "synth": "synthesizer",
    "piano": "piano playing",
    "hi-hats": "hi-hat cymbals",
    "kick": "kick drum",
    "snare": "snare drum",
    "strings": "string ensemble",
    "crowd": "crowd cheering",
    "fx": "sound effects",
}

_sam_cache: dict[str, Any] = {}
_sam_lock = __import__("threading").Lock()


def _unload_sam_audio() -> None:
    """Unload SAM Audio model from memory."""
    _sam_cache.clear()


# Register with GPU manager
try:
    from .gpu import gpu_manager as _gpu_sam
    _gpu_sam.register_unloader("sam_audio", _unload_sam_audio)
except Exception:
    pass


def _get_sam_model(model_size: str = "large"):
    """Cached SAM Audio model (loaded once per size)."""
    if model_size in _sam_cache:
        return _sam_cache[model_size]
    with _sam_lock:
        if model_size in _sam_cache:
            return _sam_cache[model_size]
        try:
            from .gpu import gpu_manager
            gpu_manager._ensure_owner("sam_audio")
        except Exception:
            pass
        _ensure_torchcodec_importable()
        from sam_audio import SAMAudio, SAMAudioProcessor  # type: ignore[import-untyped]
        import torch

        model_id = f"facebook/sam-audio-{model_size}"
        log.info("Loading SAM Audio %s from %s", model_size, model_id)
        model = SAMAudio.from_pretrained(model_id).eval()
        if torch.cuda.is_available():
            model = model.half().cuda()
        processor = SAMAudioProcessor.from_pretrained(model_id)
        _sam_cache[model_size] = (model, processor)
        return model, processor


def _separate_sam(
    path: Path,
    description: str,
    model_size: str = "large",
    reranking_candidates: int = 4,
) -> dict[str, np.ndarray]:
    """Separate a single target described by *description* using SAM Audio.

    Returns ``{"target": array, "residual": array}`` — both (samples, channels).
    """
    import torch

    model, processor = _get_sam_model(model_size)

    # Pre-load audio as tensor to avoid torchcodec dependency in torchaudio.load
    target_sr = processor.audio_sampling_rate  # 48000
    audio_np, file_sr = sf.read(str(path), dtype="float32")
    if audio_np.ndim == 1:
        audio_np = audio_np[:, np.newaxis]
    # (samples, channels) → (channels, samples) for torch
    audio_t = torch.from_numpy(audio_np.T)
    # Resample if needed
    if file_sr != target_sr:
        import torchaudio.functional as F  # type: ignore[import-untyped]
        audio_t = F.resample(audio_t, file_sr, target_sr)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # SAM Audio was trained on 30s clips.  Process longer audio in
    # overlapping chunks to avoid OOM on the DAC codec's conv pass.
    chunk_sec = 30
    overlap_sec = 5
    chunk_samples = chunk_sec * target_sr
    overlap_samples = overlap_sec * target_sr
    hop = chunk_samples - overlap_samples
    total_samples = audio_t.shape[-1]

    if total_samples <= chunk_samples:
        chunks = [(0, total_samples)]
    else:
        chunks = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunks.append((start, end))
            if end == total_samples:
                break
            start += hop

    target_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []

    for i, (s, e) in enumerate(chunks):
        log.info("SAM Audio chunk %d/%d  [%.1fs – %.1fs]",
                 i + 1, len(chunks), s / target_sr, e / target_sr)
        chunk = audio_t[:, s:e]
        batch = processor(audios=[chunk], descriptions=[description])
        batch = batch.to(device)
        if hasattr(batch, 'audios') and batch.audios is not None:
            batch.audios = batch.audios.to(dtype)

        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=False,
                reranking_candidates=reranking_candidates,
            )
        tgt = result.target if isinstance(result.target, torch.Tensor) else result.target[0]
        res = result.residual if isinstance(result.residual, torch.Tensor) else result.residual[0]
        target_parts.append(tgt.cpu())
        residual_parts.append(res.cpu())

    # Crossfade overlapping regions
    def _crossfade_chunks(parts: list[torch.Tensor]) -> torch.Tensor:
        if len(parts) == 1:
            return parts[0]
        # Normalise to 2D (channels, samples)
        normed: list[torch.Tensor] = []
        for p in parts:
            while p.dim() > 2:
                p = p.squeeze(0)
            if p.dim() == 1:
                p = p.unsqueeze(0)
            normed.append(p)
        out_parts: list[torch.Tensor] = []
        half_ol = overlap_samples // 2
        for i, part in enumerate(normed):
            chunk_len = part.shape[-1]
            if i == 0:
                out_parts.append(part[..., :chunk_len - half_ol])
            elif i == len(normed) - 1:
                out_parts.append(part[..., half_ol:])
            else:
                out_parts.append(part[..., half_ol:chunk_len - half_ol])
        return torch.cat(out_parts, dim=-1)

    target_full = _crossfade_chunks(target_parts)
    residual_full = _crossfade_chunks(residual_parts)

    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        arr = t.float().numpy()
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim == 3:
            arr = arr.squeeze(0).T
        elif arr.ndim == 2:
            arr = arr.T
        return arr

    return {
        "target": _to_numpy(target_full),
        "residual": _to_numpy(residual_full),
    }


# ---------------------------------------------------------------------------
# Demucs separation (fallback / multi-stem)
# ---------------------------------------------------------------------------

_demucs_cache: dict[str, Any] = {}
_demucs_lock = __import__("threading").Lock()


def _get_demucs_separator(model: str):
    """Cached Demucs separator (loaded once per model name)."""
    if model in _demucs_cache:
        return _demucs_cache[model]
    with _demucs_lock:
        if model in _demucs_cache:
            return _demucs_cache[model]
        import demucs.api  # type: ignore[import-untyped]
        sep = demucs.api.Separator(model=model)
        _demucs_cache[model] = sep
        return sep


def _separate_demucs(
    path: Path,
    model: str = "htdemucs",
) -> dict[str, np.ndarray]:
    """Separate using Demucs (4 or 6 stems)."""
    separator = _get_demucs_separator(model)
    origin, separated = separator.separate_audio_file(str(path))

    stems: dict[str, np.ndarray] = {}
    for name, tensor in separated.items():
        arr = tensor.numpy()
        if arr.ndim == 2:
            arr = arr.T  # (channels, samples) → (samples, channels)
        stems[name] = arr

    return stems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_stems(
    path: str | Path,
    model: str = "auto",
    quality: str = "balanced",
) -> dict[str, np.ndarray]:
    """Separate a track into stems.

    Parameters
    ----------
    model : str
        - ``"auto"``  — pick best engine by *quality* preset (default)
        - ``"roformer"``, ``"bs_roformer"``, ``"mel_band"`` — specific Roformer variant
        - ``"demucs"``, ``"htdemucs"``, ``"htdemucs_ft"``, ``"htdemucs_6s"`` — Demucs variant
        - a full ``.ckpt`` / ``.pth`` filename — passed through to audio-separator
    quality : str
        Used only when ``model="auto"``:
        - ``"fast"``     : HTDemucs 4-stem (~10 s/track on RTX 3090)
        - ``"balanced"`` : BS-RoFormer 2-stem (default, ~25 s/track)
        - ``"best"``     : BS-RoFormer + Mel-Band ensemble (~50 s/track, +0.3 dB)

    Returns dict of stem_name → numpy array (samples, channels).
    """
    path = Path(path)

    if model == "auto":
        if quality == "fast" and _has_demucs():
            try:
                return _separate_demucs(path, "htdemucs")
            except Exception:
                pass
        if quality == "best" and _has_audio_separator():
            try:
                return _separate_ensemble(path)
            except Exception:
                pass
        # balanced (default)
        if _has_audio_separator():
            try:
                return _separate_roformer(path, ROFORMER_BS_MODEL)
            except Exception:
                pass  # fall through
        if _has_demucs():
            try:
                return _separate_demucs(path)
            except Exception:
                pass
        raise ImportError(
            "No stem separation engine found or all engines failed. Install: "
            "pip install audio-separator[cpu]  OR  pip install demucs"
        )

    # Explicit model selection
    if model in ("roformer", "bs_roformer"):
        return _separate_roformer(path, ROFORMER_BS_MODEL)
    if model == "mel_band":
        return _separate_roformer(path, ROFORMER_MEL_BAND_MODEL)

    if model.endswith(".ckpt") or model.endswith(".pth"):
        return _separate_roformer(path, model)

    # SAM Audio text-prompted separation
    if model in ("sam", "sam_audio"):
        desc = SAM_PROMPT_MAP.get(quality, quality)
        return _separate_sam(path, desc)
    if model.startswith("sam-"):
        # e.g. "sam-base", "sam-small"
        size = model.split("-", 1)[1]
        desc = SAM_PROMPT_MAP.get(quality, quality)
        return _separate_sam(path, desc, model_size=size)

    # Demucs models (htdemucs, htdemucs_ft, htdemucs_6s)
    demucs_model = "htdemucs" if model == "demucs" else model
    return _separate_demucs(path, demucs_model)


def _separate_ensemble(path: Path) -> dict[str, np.ndarray]:
    """Run BS-RoFormer + Mel-Band RoFormer and average the outputs.

    Yields ~0.3 dB higher SDR than either model alone at 2× the runtime cost.
    Use only when the extra quality is worth the time (single-track mashup
    workflows, not 1000+ batch jobs).
    """
    bs = _separate_roformer(path, ROFORMER_BS_MODEL)
    mb = _separate_roformer(path, ROFORMER_MEL_BAND_MODEL)

    # Align length (models may produce slightly different tail lengths)
    result: dict[str, np.ndarray] = {}
    for key in bs.keys():
        if key not in mb:
            result[key] = bs[key]
            continue
        a, b = bs[key], mb[key]
        n = min(len(a), len(b))
        result[key] = (a[:n] + b[:n]) / 2.0
    return result


def export_stems(
    path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> list[Path]:
    """Separate and save all stems as WAV files.

    Returns list of paths to the saved stem files.
    """
    path = Path(path)
    if output_dir is None:
        output_dir = path.parent / f"{path.stem}_stems"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = separate_stems(path, model=model, quality=quality)
    saved: list[Path] = []

    for name, audio in stems.items():
        out_path = output_dir / f"{name}.wav"
        sf.write(str(out_path), audio, sr, subtype="PCM_24")
        saved.append(out_path)

    return saved


def _split_vocal_instrumental(
    path: Path, model: str, quality: str
) -> dict[str, np.ndarray]:
    """Run one stem-separation pass and return both vocals and instrumental.

    BS-RoFormer is a 2-stem model — a single inference yields both halves,
    so splitting the acapella and the instrumental from the same call
    halves the work vs running the model twice.
    """
    stems = separate_stems(path, model=model, quality=quality)

    def _first_match(*keys: str) -> np.ndarray | None:
        for k in keys:
            v = stems.get(k)
            if v is not None:
                return v
        return None

    vocals = _first_match("vocals", "voice")
    inst = _first_match("instrumental", "other", "no_vocals")
    # If we got a 4-stem output (Demucs), sum the non-vocal stems ourselves
    if inst is None:
        non_vocal = [a for k, a in stems.items() if k != "vocals"]
        if not non_vocal:
            raise ValueError("No non-vocal stems found")
        # Align lengths before summing
        n = min(len(a) for a in non_vocal)
        inst = sum(a[:n] for a in non_vocal)
    if vocals is None:
        raise ValueError("No vocal stem found")
    return {"vocals": vocals, "instrumental": inst}


def _write_normalized(audio: np.ndarray, out_path: Path, sr: int) -> Path:
    """Write audio to WAV with peak normalisation to avoid clipping."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(str(out_path), audio, sr, subtype="PCM_24")
    return out_path


def create_instrumental(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> Path:
    """Create an instrumental version (everything minus vocals).

    Default ``balanced`` quality uses BS-RoFormer 2-stem which yields a
    cleaner instrumental than summing 4 Demucs stems (where per-stem
    leakage compounds on the sum).
    """
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_instrumental")
    return _write_normalized(
        _split_vocal_instrumental(path, model, quality)["instrumental"],
        Path(output_path),
        sr,
    )


def create_acapella(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> Path:
    """Create a vocal-only version (acapella).

    Default ``balanced`` uses BS-RoFormer — top-scoring vocal model on the
    2026 MVSep multisong leaderboard.
    """
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_acapella")
    return _write_normalized(
        _split_vocal_instrumental(path, model, quality)["vocals"],
        Path(output_path),
        sr,
    )


def create_acapella_and_instrumental(
    path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> tuple[Path, Path]:
    """Produce both acapella and instrumental from a single separation pass.

    Prefer this over calling :func:`create_acapella` and
    :func:`create_instrumental` back-to-back — both share one BS-RoFormer
    inference so this is ~2× faster.
    """
    path = Path(path)
    if output_dir is None:
        output_dir = path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = _split_vocal_instrumental(path, model, quality)
    aca = _write_normalized(
        split["vocals"], output_dir / f"{path.stem}_acapella.wav", sr,
    )
    inst = _write_normalized(
        split["instrumental"], output_dir / f"{path.stem}_instrumental.wav", sr,
    )
    return aca, inst


# ---------------------------------------------------------------------------
# SAM Audio — text-prompted separation (public API)
# ---------------------------------------------------------------------------

def separate_with_prompt(
    path: str | Path,
    description: str,
    output_dir: str | Path | None = None,
    model_size: str = "large",
    sr: int = 44100,
    reranking_candidates: int = 4,
) -> tuple[Path, Path]:
    """Isolate any sound described by *description* using SAM Audio.

    Returns ``(target_path, residual_path)`` — the isolated sound and
    everything else, saved as 24-bit WAV.

    *description* can be a short DJ-friendly name (``"vocals"``, ``"synth"``,
    ``"hi-hats"``) which is mapped to a SAM-friendly prompt, or a free-text
    description (``"crowd cheering in the background"``).

    Parameters
    ----------
    model_size : str
        ``"large"`` (3 B, ~16 GB VRAM), ``"base"`` (1 B, ~8 GB), or
        ``"small"`` (500 M, ~4 GB).
    reranking_candidates : int
        Higher = better quality, slower. 1 is fastest, 8 is best.
    """
    if not _has_sam_audio():
        raise ImportError(
            "SAM Audio not installed. Install: pip install sam-audio "
            "and authenticate: huggingface-cli login"
        )

    path = Path(path)
    prompt = SAM_PROMPT_MAP.get(description.lower(), description)

    if output_dir is None:
        output_dir = path.parent / f"{path.stem}_stems"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = _separate_sam(
        path, prompt, model_size=model_size,
        reranking_candidates=reranking_candidates,
    )

    # Sanitise description for filename
    safe_name = description.lower().replace(" ", "_")[:30]
    target_path = _write_normalized(
        stems["target"], output_dir / f"{safe_name}.wav", sr,
    )
    residual_path = _write_normalized(
        stems["residual"], output_dir / f"{safe_name}_residual.wav", sr,
    )
    return target_path, residual_path
