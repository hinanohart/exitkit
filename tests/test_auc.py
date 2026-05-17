"""Toy synthetic dataset to verify the continuity metric is discriminative.

We generate 50 positive pairs (same agent, ~15% memories perturbed) and 50
negative pairs (two unrelated agents drawn from disjoint vocabularies) and
require ROC-AUC >= 0.7 between the two populations of continuity scores.
"""

from __future__ import annotations

import random

from portable_ai_memory import MemoryObject, MemoryStore, Owner

from exitkit import continuity_score

_VOCAB_A = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
]
_VOCAB_B = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "black",
    "white",
    "brown",
    "cyan",
    "magenta",
    "silver",
    "gold",
    "bronze",
    "ivory",
]


def _make_store(
    rng: random.Random,
    vocab: list[str],
    n: int,
    prefix: str,
) -> MemoryStore:
    memories = [
        MemoryObject.create(
            id=f"{prefix}-{i}",
            type="fact",
            content=" ".join(rng.choices(vocab, k=rng.randint(4, 8))),
            platform="toy",
        )
        for i in range(n)
    ]
    return MemoryStore(schema_version="1.0", owner=Owner(id="toy"), memories=memories)


def _perturb(
    rng: random.Random,
    store: MemoryStore,
    vocab: list[str],
    frac: float,
) -> MemoryStore:
    new_memories: list[MemoryObject] = []
    for mem in store.memories:
        if rng.random() < frac:
            words = mem.content.split()
            if words:
                idx = rng.randrange(len(words))
                words[idx] = rng.choice(vocab)
                new_content = " ".join(words)
            else:
                new_content = rng.choice(vocab)
            new_memories.append(
                MemoryObject.create(
                    id=mem.id,
                    type="fact",
                    content=new_content,
                    platform="toy",
                )
            )
        else:
            new_memories.append(mem)
    return MemoryStore(schema_version="1.0", owner=Owner(id="toy"), memories=new_memories)


def _auc(positives: list[float], negatives: list[float]) -> float:
    """Mann-Whitney rank-statistic ROC-AUC."""
    wins = 0
    ties = 0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def test_toy_dataset_auc_above_threshold() -> None:
    rng = random.Random(0xEC17)
    n_pairs = 50

    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for _ in range(n_pairs):
        base = _make_store(rng, _VOCAB_A, n=20, prefix="pos-base")
        perturbed = _perturb(rng, base, _VOCAB_A, frac=0.15)
        positive_scores.append(continuity_score(base, perturbed).continuity)

        agent_a = _make_store(rng, _VOCAB_A, n=20, prefix="other-a")
        agent_b = _make_store(rng, _VOCAB_B, n=20, prefix="other-b")
        negative_scores.append(continuity_score(agent_a, agent_b).continuity)

    auc = _auc(positive_scores, negative_scores)
    assert auc >= 0.7, f"AUC {auc:.3f} < 0.7 threshold"
