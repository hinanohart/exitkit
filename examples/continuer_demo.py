"""Minimal demo of `exitkit.continuity_score` over two PAM snapshots.

Threshold the printed continuity (e.g. ``>= 0.8``) to use the score as a
drift binary classifier for fine-tunes, migrations, or personalisation
passes — the Nozickian closest-continuer reading of the after-snapshot.
"""

from __future__ import annotations

from portable_ai_memory import MemoryObject, MemoryStore, Owner

from exitkit import continuity_score


def _make(memories: list[tuple[str, str]]) -> MemoryStore:
    return MemoryStore(
        schema_version="1.0",
        owner=Owner(id="demo-user"),
        memories=[
            MemoryObject.create(id=mid, type="fact", content=content, platform="demo")
            for mid, content in memories
        ],
    )


def main() -> None:
    before = _make(
        [
            ("m1", "Prefers async over threads."),
            ("m2", "Lives in Tokyo."),
            ("m3", "Likes minimalist UIs."),
        ]
    )
    after = _make(
        [
            ("m1", "Prefers async over threads, except for I/O-bound work."),
            ("m2", "Lives in Tokyo."),
            ("m4", "Working on the ExitKit OSS."),
        ]
    )
    report = continuity_score(before, after)
    print(f"continuity:     {report.continuity:.3f}")
    print(f"identity diff:  {report.identity_diff:.3f}")
    print(f"semantic drift: {report.semantic_drift:.3f}")
    print(f"added:    {sorted(report.added)}")
    print(f"removed:  {sorted(report.removed)}")
    print(f"mutated:  {sorted(report.mutated)}")
    print(f"weights:  {report.weights}")
    print(f"n_before: {report.n_before}  n_after: {report.n_after}")


if __name__ == "__main__":
    main()
