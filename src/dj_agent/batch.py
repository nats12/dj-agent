"""V3 batch processing — concurrent CPU/GPU pipeline for library-scale analysis.

Overlaps CPU-bound work (librosa features, LUFS, beat tracking) with
GPU-bound work (Essentia TF key/mood/vocals) using a thread pool.

Also batches all Essentia TF models into a single audio load per track
instead of loading the file 3 separate times.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import get_config
from .memory import load_memory, save_memory, store_track_analysis, get_track_analysis

_log = logging.getLogger(__name__)


def analyse_library_batch(
    tracks: list[Path],
    memory: dict[str, Any] | None = None,
    workers: int = 2,
    save_every: int = 25,
    skip_cached: bool = True,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Analyse a library of tracks with concurrent CPU/GPU processing.

    Parameters
    ----------
    tracks : list of audio file paths
    memory : memory dict (loaded via load_memory). If None, loads from config.
    workers : number of concurrent analysis threads (2 = overlap CPU and GPU)
    save_every : save memory to disk every N tracks
    skip_cached : skip tracks already in memory
    progress_callback : optional callable(done, total, track_name, result)

    Returns dict with summary stats.
    """
    config = get_config()
    if memory is None:
        memory = load_memory(config.memory)

    total = len(tracks)
    done = 0
    errors = 0
    skipped = 0
    t0 = time.time()

    _log.info("Batch analysis: %d tracks, %d workers", total, workers)

    # Pre-warm Essentia (loads TF models once)
    _prewarm_essentia()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}

        for i, track_path in enumerate(tracks):
            # Check cache
            if skip_cached:
                cached = get_track_analysis(memory, track_path)
                if cached:
                    skipped += 1
                    done += 1
                    if progress_callback:
                        progress_callback(done, total, track_path.name, cached)
                    continue

            future = pool.submit(_analyse_one, track_path)
            futures[future] = (i, track_path)

        for future in as_completed(futures):
            i, track_path = futures[future]
            done += 1

            try:
                result = future.result()
                store_track_analysis(memory, track_path, result)

                if result.get("errors"):
                    errors += 1
            except Exception as e:
                errors += 1
                _log.warning("Failed: %s — %s", track_path.name, e)
                result = {"errors": [str(e)]}

            if progress_callback:
                progress_callback(done, total, track_path.name, result)

            # Incremental save
            if done % save_every == 0:
                save_memory(memory, config.memory)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                _log.info(
                    "[%d/%d] %.1f/s | %d stored | %d err | ETA %.0fm",
                    done, total, rate, len(memory["processed_tracks"]), errors, eta,
                )

    # Final save
    save_memory(memory, config.memory)
    elapsed = time.time() - t0

    stats = {
        "total": total,
        "processed": done - skipped,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(elapsed, 1),
        "tracks_per_sec": round(done / elapsed, 2) if elapsed > 0 else 0,
        "stored": len(memory["processed_tracks"]),
    }
    _log.info("Batch complete: %s", stats)
    return stats


def _analyse_one(path: Path) -> dict[str, Any]:
    """Analyse a single track using the pipeline."""
    from .pipeline import analyse_track_full
    return analyse_track_full(path, memory=None, skip_if_cached=False)


# ---------------------------------------------------------------------------
# Essentia batch — single audio load for key + mood + vocals
# ---------------------------------------------------------------------------

_essentia_warmed = False


def _prewarm_essentia() -> None:
    """Pre-load Essentia TF models so they're cached for the batch."""
    global _essentia_warmed
    if _essentia_warmed:
        return
    try:
        import essentia.standard as es
        # Touch the key extractor to trigger model caching
        if hasattr(es, "KeyExtractor"):
            es.KeyExtractor(profileType="edma")
            _log.info("Essentia EDMA key extractor pre-warmed")
        _essentia_warmed = True
    except Exception:
        pass


# Cached Essentia objects (loaded once, reused across tracks)
_tf_predictors: dict[str, Any] = {}
_es_loaders: dict[str, Any] = {}


def _get_tf_predictor(name: str) -> Any:
    """Get or create a cached Essentia TF predictor."""
    if name in _tf_predictors:
        return _tf_predictors[name]

    import os
    import essentia.standard as es
    _models_dir = os.path.expanduser("~/.essentia/models")

    if name == "embeddings":
        pred = es.TensorflowPredictMusiCNN(
            graphFilename=os.path.join(_models_dir, "msd-musicnn-1.pb"),
            output="model/dense/BiasAdd",
        )
    elif name == "vocal":
        pred = es.TensorflowPredict2D(
            graphFilename=os.path.join(_models_dir, "voice_instrumental-msd-musicnn-1.pb"),
            output="model/Softmax",
        )
    else:
        # Mood model
        pred = es.TensorflowPredict2D(
            graphFilename=os.path.join(_models_dir, f"mood_{name}-msd-musicnn-1.pb"),
            output="model/Softmax",
        )

    _tf_predictors[name] = pred
    return pred


def _load_audio_essentia(path: str, sr: int) -> Any:
    """Load audio with a cached MonoLoader (reconfigured per file)."""
    import essentia.standard as es
    key = str(sr)
    if key not in _es_loaders:
        _es_loaders[key] = es.MonoLoader(filename=path, sampleRate=sr)
    else:
        _es_loaders[key].configure(filename=path, sampleRate=sr)
    return _es_loaders[key]()


def analyse_essentia_batch(
    path: str | Path,
) -> dict[str, Any]:
    """Run ALL Essentia analyses on one track in a single audio load.

    Returns dict with key, mood, vocal results — avoiding 3 separate
    file loads that the individual modules would do.
    TF predictors and MonoLoaders are cached so model/loader init is once only.
    """
    result: dict[str, Any] = {}

    try:
        import essentia.standard as es

        # Single load at 44100Hz (Essentia's native rate)
        audio = _load_audio_essentia(str(path), 44100)

        # Key detection (EDMA profile for electronic music)
        if hasattr(es, "KeyExtractor"):
            key_extractor = es.KeyExtractor(profileType="edma")
            key, scale, strength = key_extractor(audio)
            result["key"] = f"{key} {scale}"
            result["key_confidence"] = float(strength)
            result["key_method"] = "essentia"

        # Mood + Vocals via TF models (if available)
        if hasattr(es, "TensorflowPredictMusiCNN"):
            audio_16k = _load_audio_essentia(str(path), 16000)

            try:
                import numpy as np

                # Shared embeddings (cached predictor)
                embeddings = _get_tf_predictor("embeddings")(audio_16k)

                # Mood classifiers (cached predictors)
                scores = {}
                for mood_name in ("aggressive", "happy", "party", "relaxed", "sad"):
                    try:
                        preds = _get_tf_predictor(mood_name)(embeddings)
                        scores[mood_name] = float(np.mean(preds[:, 1]))
                    except Exception:
                        pass

                if scores:
                    result["mood"] = max(scores, key=scores.get)
                    result["mood_scores"] = scores
                    result["mood_method"] = "essentia"

                # Vocal/Instrumental (cached predictor)
                try:
                    vocal_preds = _get_tf_predictor("vocal")(embeddings)
                    mean_preds = np.mean(vocal_preds, axis=0)
                    vocal_prob = float(mean_preds[1]) if len(mean_preds) > 1 else 0.5
                    result["vocal_probability"] = vocal_prob
                    result["vocal_classification"] = (
                        "instrumental" if vocal_prob < 0.25
                        else "partial_vocal" if vocal_prob < 0.6
                        else "vocal"
                    )
                    result["vocal_method"] = "essentia"
                except Exception:
                    pass

            except Exception as e:
                _log.debug("Essentia TF models unavailable: %s", e)

    except ImportError:
        pass

    return result
