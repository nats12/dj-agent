---
name: find-similar
description: Find tracks that sound similar using audio feature embeddings and cosine similarity.
---

# Find Similar

Given a track, find the most sonically similar tracks in your library using audio feature analysis.

## Usage

```python
from dj_agent.similarity import compute_feature_vector, find_similar, build_embedding_cache

# Build cache (one-time, ~30 min for 1000 tracks)
cache = build_embedding_cache(tracks, cache_path="~/.dj-agent/embeddings.npz")

# Find similar
target = compute_feature_vector("/path/to/track.flac")
results = find_similar(target, cache, top_k=10)
```

## Features Used

MFCCs (timbre), chroma (harmonic content), spectral contrast, spectral centroid, rolloff, zero-crossing rate — 62-dimensional feature vector per track.

## Workflow

1. `find similar to [track]` — show 10 most similar tracks
2. Useful for discovering connections in your library you didn't know about
