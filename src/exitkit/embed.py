"""Default lightweight embedder for ExitKit.

Pure-numpy hashing-trick bag-of-words. Reproducible across runs and
platforms because blake2b is deterministic. Replace with sentence-
transformers or any other model by passing ``embedder=...`` to
:func:`exitkit.continuity_score`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _hash_token(token: str, n_features: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_features


def hashing_embedder(
    texts: Iterable[str],
    *,
    n_features: int = 1024,
) -> np.ndarray:
    """Embed an iterable of strings into an ``(N, n_features)`` L2-normalised matrix.

    Empty input returns a ``(0, n_features)`` zero matrix. Empty strings are
    embedded as zero rows.
    """
    text_list = list(texts)
    if not text_list:
        return np.zeros((0, n_features), dtype=np.float64)
    mat = np.zeros((len(text_list), n_features), dtype=np.float64)
    for i, text in enumerate(text_list):
        for token in _tokenize(text):
            mat[i, _hash_token(token, n_features)] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalised: np.ndarray = mat / norms
    return normalised
