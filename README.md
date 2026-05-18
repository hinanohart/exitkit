# ExitKit

> A Nozickian closest-continuer metric for portable agent memory.
> Given two [PAM](https://github.com/portable-ai-memory) snapshots of an agent (before and after a fine-tune, migration, or personalisation pass), ExitKit returns a continuity score in `[0, 1]` and a structured drift report — so you can answer *"is it still the same agent?"* with a number instead of vibes.

[![CI](https://github.com/hinanohart/exitkit/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/exitkit/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](#installation)

## Why

LLM agents accumulate memory: preferences, facts, project context. As you fine-tune, switch providers, or run a personalisation pass, the memory mutates. Did the agent stay the same agent, or did you replace it?

Robert Nozick's *Tracking Truth* and *closest-continuer* framework (Philosophical Explanations, 1981, §1) gives a principled answer: the post-update snapshot is the continuer iff it remains the highest-scoring candidate by a continuity metric and exceeds a chosen threshold.

ExitKit ports that idea to PAM-format memory snapshots and returns:

- a deterministic continuity score in `[0, 1]`,
- the `added` / `removed` / `mutated` memory IDs (content-hash aware),
- the underlying `identity_diff` and `semantic_drift` components,
- the weights used (default 0.5 / 0.5, fully configurable).

## Installation

```bash
pip install exitkit
```

Requires Python 3.11+.

## Quick start

```python
from portable_ai_memory import MemoryObject, MemoryStore, Owner
from exitkit import continuity_score


def make(memories):
    return MemoryStore(
        schema_version="1.0",
        owner=Owner(id="alice"),
        memories=[
            MemoryObject.create(id=mid, type="fact", content=c, platform="demo")
            for mid, c in memories
        ],
    )


before = make([("m1", "Prefers async."), ("m2", "Lives in Tokyo.")])
after  = make([("m1", "Prefers async, except for I/O-bound."), ("m3", "Working on ExitKit.")])

report = continuity_score(before, after)
print(report.continuity)             # 0.0 - 1.0  (1.0 = identical)
print(report.added, report.removed, report.mutated)
```

See [`examples/continuer_demo.py`](examples/continuer_demo.py) for a runnable end-to-end demo.

## What the metric does

`continuity_score(before, after, *, identity_weight=0.5, semantic_weight=0.5, embedder=None)` returns a `DriftReport`:

| Field             | Meaning                                                                          |
|-------------------|----------------------------------------------------------------------------------|
| `continuity`      | `1 - (w_id * identity_diff + w_sem * semantic_drift)`, clipped to `[0, 1]`       |
| `identity_diff`   | Content-hash-aware symmetric difference over MemoryObject IDs.                   |
| `semantic_drift`  | `1 - cosine(centroid(before), centroid(after))`, rescaled to `[0, 1]`.           |
| `added`           | `frozenset[str]` of new memory IDs.                                              |
| `removed`         | `frozenset[str]` of dropped memory IDs.                                          |
| `mutated`         | `frozenset[str]` of kept-but-rewritten memory IDs (same ID, new `content_hash`). |
| `weights`         | The `(identity_weight, semantic_weight)` actually used.                          |
| `n_before`, `n_after` | Memory counts.                                                               |

Weights must each lie in `[0, 1]` and sum to 1.

### Custom embedder

The default semantic component uses a dependency-light hashing bag-of-words (pure numpy, deterministic). For richer signals, pass any `Callable[[Iterable[str]], np.ndarray]`:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embed = lambda texts: np.asarray(model.encode(list(texts)))

report = continuity_score(before, after, embedder=embed)
```

## Design notes

- **Use it as a drift binary classifier.** Threshold the `continuity` score (e.g. `>= 0.8`) to flag whether a fine-tune, migration, or personalisation pass kept the agent's memory identity intact — the toy benchmark in `tests/test_auc.py` shows the default weights are discriminative (ROC-AUC ≥ 0.7 against unrelated agents).
- **`semantic_drift` range depends on the embedder.** With the default `hashing_embedder` (non-negative bag-of-words), cosines lie in `[0, 1]`, so `semantic_drift` is bounded by `[0, 0.5]` for non-empty stores. Pass a sentence-transformers (or other arbitrary-direction) embedder if you need the full `[0, 1]` range. Empty vs. empty (drift = 0.0) and empty vs. non-empty (drift = 1.0) remain reachable under any embedder.
- **Weights are applied to drift, not to a normalised scale.** `identity_diff` always spans `[0, 1]`, but with the default embedder `semantic_drift` only spans `[0, 0.5]`. The default `0.5 / 0.5` weighting therefore puts roughly twice the effective weight on the identity component. Pass an embedder that spans the full cosine range, or set `identity_weight` / `semantic_weight` explicitly, if that asymmetry matters for your use case.
- **MemoryObject IDs must be unique per store.** `continuity_score` raises `ValueError` if a `MemoryStore` contains duplicate IDs — silently collapsing duplicates produced subtle false-positive `mutated` results.
- **Default tokenisation is alphanumeric (`\w+`, unicode).** Memories whose content is only punctuation, whitespace, or emoji collapse to zero tokens under `hashing_embedder` and therefore contribute no semantic signal. Pass a richer custom embedder if those signals matter.
- **One component, on purpose.** v0.1 is the *continuer-select* metric only — no UI, no provenance store, no policy engine. Cedar-based export policies and Sigstore-signed manifests are tracked for v0.2.
- **Tracking Truth ≠ aggregation.** ExitKit does not try to aggregate users or vote on values; it measures whether a single agent's memory state continues, in the closest-continuer sense.
- **Convergent goals, not endorsement.** The "agent stays the agent you trained" objective overlaps with the *positive alignment* programme described in Laukkonen et al., *Positive Alignment* (arXiv:2605.10310, 2026); cited as a convergent vision, not a methodological commitment.

## Development

```bash
git clone https://github.com/hinanohart/exitkit
cd exitkit
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

## References

- Santhosh Kumar Ravindran. *Portable Agent Memory: A Protocol for Provenance-Verified Memory Transfer Across Heterogeneous LLM Agents.* arXiv:2605.11032 (2026).
- Robert Nozick. *Anarchy, State, and Utopia.* Basic Books, 1974 — Part III, "A Framework for Utopia".
- Robert Nozick. *Philosophical Explanations.* Harvard University Press, 1981 — §1 Tracking Truth and §1 closest-continuer.
- Ruben Laukkonen, Seb Krier, Chloé Bakalar et al. *Positive Alignment: Artificial Intelligence for Human Flourishing.* arXiv:2605.10310 (2026).

## License

MIT License — see [LICENSE](LICENSE).
