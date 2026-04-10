# Known Limitations

Documented tradeoffs and operational risks in dj_agent v2.2.
These are intentional design decisions, not bugs.

## Reasoning Module

- **Backend detection is proxy-based, not readiness-based.** `_flamingo_available()` checks CUDA + imports, not model inference. `_gemini_available()` checks API key existence, not API reachability. First real call may still fail.
- **Temp file cleanup is age-gated, not session-aware.** Files older than 1 hour are deleted on import. A long-running analysis (>1 hour) in another process could have its temp files deleted. Mitigation: analysis typically takes <2 minutes per track.
- **Snippet sampling at 25% and 50% is a heuristic.** Progressive tracks with late drops (>60%) may be underrepresented. The offset_pct parameter allows callers to override.

## Similarity

- **Dimension mismatch returns 0.0 similarity, not an error.** This conflates "incompatible vectors" with "orthogonal vectors." The cache build locks method after the first track to prevent mixing, but pre-existing caches from a different method are silently treated as dissimilar.

## Phrase Detection

- **All-In-One label mapping (chorus→drop, verse→build) is unvalidated.** Pop-song structure labels are not the same as DJ transition semantics. Works for 4/4 electronic music, less reliable for ambient, breaks, or jazz-influenced styles.

## Set Building

- **The transition scorer is a weighted heuristic, not a learned model.** Genre is binary (same/different), missing energy defaults to 0.5, BPM/key are coarsely binned. Produces reasonable orderings but is not a research-grade optimizer.

## GPU Management

- **The GPUManager is coordination by convention, not hard enforcement.** Models must call `_ensure_owner()` voluntarily. Third-party libraries (Demucs, Roformer) may allocate VRAM outside the manager's view.

## Mastering

- **Idempotency guard uses filename pattern (`_mastered`), not content hashing.** Renamed files can bypass the guard. A hash-based approach would require a separate registry database.

## Metadata

- **Enrichment is first-hit, best-effort.** MusicBrainz returns the first recording match. Last.fm filters by hardcoded keyword lists. Discogs takes the first release result. No Beatport implementation exists.
