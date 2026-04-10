"""Set list optimization — graph-based track ordering with energy arc constraints.

Solves a variant of the Traveling Salesman Problem (TSP):
- ≤20 tracks: greedy nearest-neighbor + 2-opt local search (fast, near-optimal)
- Constraint: energy arc (warmup → build → peak → cooldown)
"""

from __future__ import annotations

import random
from typing import Any

from .harmonic import score_transition as harmonic_score
from .types import TrackInfo


# ---------------------------------------------------------------------------
# Energy arc templates
# ---------------------------------------------------------------------------

ENERGY_ARCS = {
    "warmup_to_peak": [3, 4, 5, 6, 7, 8, 9, 10, 10, 9],
    "peak_time": [7, 8, 9, 10, 10, 9, 10, 10, 9, 8],
    "chill_set": [3, 4, 5, 5, 4, 5, 6, 5, 4, 3],
    "full_night": [2, 3, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 5, 3],
    "flat": None,  # no energy constraint
}


def build_set(
    tracks: list[TrackInfo],
    energies: dict[str, int] | None = None,
    arc: str = "flat",
    max_iterations: int = 1000,
) -> list[TrackInfo]:
    """Order tracks for optimal transitions with optional energy arc.

    Parameters
    ----------
    tracks : list of TrackInfo
    energies : dict mapping content_id → energy (1-10)
    arc : energy arc template name (see ENERGY_ARCS)
    max_iterations : number of 2-opt improvement iterations

    Returns ordered list of TrackInfo.
    """
    if len(tracks) <= 1:
        return list(tracks)

    if len(tracks) == 2:
        return list(tracks)

    # Build distance matrix (lower = better transition)
    n = len(tracks)
    energies = energies or {}
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        e_i = energies.get(tracks[i].db_content_id, 0)
        for j in range(n):
            if i != j:
                e_j = energies.get(tracks[j].db_content_id, 0)
                score = harmonic_score(tracks[i], tracks[j],
                                       energy_a=e_i, energy_b=e_j)
                dist[i][j] = 1.0 - score  # invert: lower distance = better

    # Greedy nearest-neighbor starting from the track with lowest energy
    if energies:
        start_idx = min(
            range(n),
            key=lambda i: energies.get(tracks[i].db_content_id, 5),
        )
    else:
        start_idx = 0

    order = _greedy_nearest_neighbor(dist, start_idx, n)

    # 2-opt local search improvement
    order = _two_opt(order, dist, max_iterations)

    # Apply energy arc constraint if specified
    target_arc = ENERGY_ARCS.get(arc)
    if target_arc and energies:
        order = _apply_energy_arc(order, tracks, energies, target_arc)

    return [tracks[i] for i in order]


def _greedy_nearest_neighbor(
    dist: list[list[float]], start: int, n: int,
) -> list[int]:
    """Greedy nearest-neighbor TSP heuristic."""
    visited = {start}
    order = [start]

    current = start
    for _ in range(n - 1):
        best = -1
        best_dist = float("inf")
        for j in range(n):
            if j not in visited and dist[current][j] < best_dist:
                best = j
                best_dist = dist[current][j]
        if best == -1:
            break
        visited.add(best)
        order.append(best)
        current = best

    return order


def _two_opt(
    order: list[int],
    dist: list[list[float]],
    max_iter: int,
) -> list[int]:
    """2-opt local search to improve tour quality."""
    n = len(order)
    if n <= 3:
        return order

    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Cost of current edges
                d_old = dist[order[i]][order[i + 1]] + (
                    dist[order[j]][order[(j + 1) % n]] if j + 1 < n else 0
                )
                # Cost of swapped edges
                d_new = dist[order[i]][order[j]] + (
                    dist[order[i + 1]][order[(j + 1) % n]] if j + 1 < n else 0
                )
                if d_new < d_old - 1e-6:
                    order[i + 1: j + 1] = reversed(order[i + 1: j + 1])
                    improved = True

    return order


def _apply_energy_arc(
    order: list[int],
    tracks: list[TrackInfo],
    energies: dict[str, int],
    target_arc: list[int],
) -> list[int]:
    """Re-sort to approximate the target energy arc.

    Divides tracks into positional slots and assigns tracks whose energy
    is closest to the target for that slot, while preserving harmonic
    compatibility as a secondary factor.
    """
    n = len(order)
    # Interpolate target arc to match track count
    arc_len = len(target_arc)
    target = [
        target_arc[int(i * arc_len / n) % arc_len] for i in range(n)
    ]

    # Score each track for each position
    indexed = list(order)
    available = set(indexed)
    result: list[int] = []

    for pos in range(n):
        target_energy = target[pos]
        best_idx = -1
        best_score = float("inf")

        for idx in available:
            e = energies.get(tracks[idx].db_content_id, 5)
            energy_diff = abs(e - target_energy)

            # Harmonic penalty: prefer good transitions from previous track
            harmonic_penalty = 0.0
            if result:
                prev = result[-1]
                h_score = harmonic_score(tracks[prev], tracks[idx])
                harmonic_penalty = (1.0 - h_score) * 3  # weight harmonic less

            score = energy_diff + harmonic_penalty
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            result.append(best_idx)
            available.discard(best_idx)

    return result
