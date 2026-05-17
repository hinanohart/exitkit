"""Tracking-Truth-inspired continuity metric over PAM MemoryStore snapshots.

Given two `portable_ai_memory.MemoryStore` snapshots — for example an agent
before and after a fine-tune, migration, or personalisation pass —
:func:`continuity_score` returns a :class:`DriftReport` with a continuity
value in ``[0, 1]`` plus the underlying identity-diff and semantic-drift
components.

The metric is a weighted average of two components, both bounded to ``[0, 1]``:

* ``identity_diff`` — content-hash-aware symmetric difference over
  MemoryObject IDs (added / removed / mutated, where *mutated* = same ID,
  different ``content_hash``).
* ``semantic_drift`` — ``1 - cosine(centroid(before), centroid(after))``
  rescaled to ``[0, 1]``.

The framing follows Nozick's closest-continuer schema (Philosophical
Explanations, 1981, §1): the post-update snapshot counts as the continuer
iff its continuity score is the highest among candidates and exceeds a
caller-chosen threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exitkit.embed import hashing_embedder

if TYPE_CHECKING:
    from portable_ai_memory import MemoryStore


Embedder = Callable[[Iterable[str]], np.ndarray]


@dataclass(frozen=True)
class DriftReport:
    """Structured result of a continuity computation."""

    continuity: float
    identity_diff: float
    semantic_drift: float
    added: frozenset[str]
    removed: frozenset[str]
    mutated: frozenset[str]
    weights: tuple[float, float]
    n_before: int
    n_after: int


def _index(store: MemoryStore) -> dict[str, str]:
    return {mem.id: mem.content_hash for mem in store.memories}


def _identity_diff(
    before_idx: dict[str, str],
    after_idx: dict[str, str],
) -> tuple[float, frozenset[str], frozenset[str], frozenset[str]]:
    ids_before = set(before_idx)
    ids_after = set(after_idx)
    added = ids_after - ids_before
    removed = ids_before - ids_after
    kept = ids_before & ids_after
    mutated = {mid for mid in kept if before_idx[mid] != after_idx[mid]}
    union = ids_before | ids_after
    diff = 0.0 if not union else (len(added) + len(removed) + len(mutated)) / len(union)
    return diff, frozenset(added), frozenset(removed), frozenset(mutated)


def _semantic_drift(
    before: MemoryStore,
    after: MemoryStore,
    embedder: Embedder,
) -> float:
    before_texts = [m.content for m in before.memories]
    after_texts = [m.content for m in after.memories]
    if not before_texts and not after_texts:
        return 0.0
    if not before_texts or not after_texts:
        return 1.0
    mat_before = embedder(before_texts)
    mat_after = embedder(after_texts)
    centroid_before = np.asarray(mat_before).mean(axis=0)
    centroid_after = np.asarray(mat_after).mean(axis=0)
    nb = float(np.linalg.norm(centroid_before))
    na = float(np.linalg.norm(centroid_after))
    if nb == 0.0 and na == 0.0:
        # Both stores produced no extractable signal under this embedder; treat
        # as vacuously equal so reflexivity holds.
        return 0.0
    if nb == 0.0 or na == 0.0:
        return 1.0
    cosine = float(np.dot(centroid_before, centroid_after) / (nb * na))
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - (cosine + 1.0) / 2.0


def continuity_score(
    before: MemoryStore,
    after: MemoryStore,
    *,
    identity_weight: float = 0.5,
    semantic_weight: float = 0.5,
    embedder: Embedder | None = None,
) -> DriftReport:
    """Compute the Tracking-Truth continuity score between two PAM stores.

    Args:
        before: the prior MemoryStore snapshot.
        after: the later MemoryStore snapshot.
        identity_weight: weight on the structural identity-diff component;
            must be in ``[0, 1]``.
        semantic_weight: weight on the semantic-drift component; must be
            in ``[0, 1]``. ``identity_weight + semantic_weight`` must equal 1.
        embedder: optional callable mapping an iterable of strings to an
            ``(N, D)`` numpy matrix. Defaults to
            :func:`exitkit.embed.hashing_embedder`.

    Returns:
        DriftReport with ``continuity`` in ``[0, 1]`` (1.0 = identical).
    """
    if not 0.0 <= identity_weight <= 1.0:
        raise ValueError("identity_weight must be in [0, 1]")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be in [0, 1]")
    if abs(identity_weight + semantic_weight - 1.0) > 1e-9:
        raise ValueError("identity_weight + semantic_weight must equal 1.0")

    emb: Embedder = embedder if embedder is not None else hashing_embedder

    before_idx = _index(before)
    after_idx = _index(after)
    identity_diff, added, removed, mutated = _identity_diff(before_idx, after_idx)
    semantic_drift = _semantic_drift(before, after, emb)
    drift = identity_weight * identity_diff + semantic_weight * semantic_drift
    continuity = max(0.0, min(1.0, 1.0 - drift))

    return DriftReport(
        continuity=continuity,
        identity_diff=identity_diff,
        semantic_drift=semantic_drift,
        added=added,
        removed=removed,
        mutated=mutated,
        weights=(identity_weight, semantic_weight),
        n_before=len(before.memories),
        n_after=len(after.memories),
    )
