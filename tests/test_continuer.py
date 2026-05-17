from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from portable_ai_memory import MemoryObject, MemoryStore, Owner

from exitkit import continuity_score
from exitkit.embed import hashing_embedder


def _store(memories: list[MemoryObject]) -> MemoryStore:
    return MemoryStore(
        schema_version="1.0",
        owner=Owner(id="user-1"),
        memories=memories,
    )


def _mem(mid: str, content: str, mtype: str = "fact") -> MemoryObject:
    return MemoryObject.create(
        id=mid,
        type=mtype,
        content=content,
        platform="test",
    )


def test_identical_stores_have_continuity_one() -> None:
    s = _store([_mem("a", "hello"), _mem("b", "world")])
    report = continuity_score(s, s)
    assert report.continuity == pytest.approx(1.0)
    assert report.identity_diff == 0.0
    assert report.semantic_drift == pytest.approx(0.0, abs=1e-9)
    assert report.added == frozenset()
    assert report.removed == frozenset()
    assert report.mutated == frozenset()


def test_completely_disjoint_stores_have_low_continuity() -> None:
    before = _store([_mem("a", "the cat sat on the mat")])
    after = _store([_mem("z", "quantum chromodynamics is a gauge theory")])
    report = continuity_score(before, after)
    assert report.continuity < 0.4
    assert report.added == frozenset({"z"})
    assert report.removed == frozenset({"a"})


def test_mutation_detected_via_content_hash() -> None:
    before = _store([_mem("a", "v1")])
    after = _store([_mem("a", "v2")])
    report = continuity_score(before, after)
    assert "a" in report.mutated
    assert report.added == frozenset()
    assert report.removed == frozenset()
    assert report.continuity < 1.0


def test_weights_must_sum_to_one() -> None:
    s = _store([_mem("a", "x")])
    with pytest.raises(ValueError):
        continuity_score(s, s, identity_weight=0.3, semantic_weight=0.3)


def test_weights_must_be_in_unit_interval() -> None:
    s = _store([_mem("a", "x")])
    with pytest.raises(ValueError):
        continuity_score(s, s, identity_weight=-0.1, semantic_weight=1.1)


def test_empty_stores_continuity() -> None:
    s = _store([])
    report = continuity_score(s, s)
    assert report.continuity == pytest.approx(1.0)
    assert report.n_before == 0
    assert report.n_after == 0


def test_one_empty_one_nonempty_drifts() -> None:
    before = _store([_mem("a", "alpha")])
    after = _store([])
    report = continuity_score(before, after)
    assert report.semantic_drift == pytest.approx(1.0)
    assert report.removed == frozenset({"a"})


def test_custom_embedder_called() -> None:
    calls: list[list[str]] = []

    def fake_embedder(texts: object) -> np.ndarray:
        text_list = list(texts)  # type: ignore[arg-type]
        calls.append(text_list)
        return np.ones((len(text_list), 4))

    s1 = _store([_mem("a", "foo")])
    s2 = _store([_mem("b", "bar")])
    report = continuity_score(s1, s2, embedder=fake_embedder)
    assert calls
    assert report.semantic_drift == pytest.approx(0.0)


def test_report_carries_counts() -> None:
    before = _store([_mem("a", "x"), _mem("b", "y")])
    after = _store([_mem("a", "x")])
    report = continuity_score(before, after)
    assert report.n_before == 2
    assert report.n_after == 1


def test_default_embedder_is_l2_normalised() -> None:
    mat = hashing_embedder(["the quick brown fox", "lazy dog"])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0)


def test_default_embedder_empty_input() -> None:
    mat = hashing_embedder([])
    assert mat.shape == (0, 1024)


# ─── Hypothesis property tests (5 invariants) ─────────────────────


@st.composite
def stores_strategy(draw: st.DrawFn, *, max_size: int = 6) -> MemoryStore:
    n = draw(st.integers(min_value=0, max_value=max_size))
    id_pool = [f"m{i}" for i in range(max_size)]
    chosen_ids = draw(
        st.lists(st.sampled_from(id_pool), min_size=n, max_size=n, unique=True)
    )
    memories = [
        _mem(
            mid,
            draw(
                st.text(
                    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
                    min_size=1,
                    max_size=32,
                )
            ),
        )
        for mid in chosen_ids
    ]
    return _store(memories)


@given(stores_strategy())
@settings(max_examples=40, deadline=None)
def test_invariant_reflexive(s: MemoryStore) -> None:
    r = continuity_score(s, s)
    assert r.continuity == pytest.approx(1.0)


@given(stores_strategy(), stores_strategy())
@settings(max_examples=40, deadline=None)
def test_invariant_unit_interval(s1: MemoryStore, s2: MemoryStore) -> None:
    r = continuity_score(s1, s2)
    assert 0.0 <= r.continuity <= 1.0
    assert 0.0 <= r.identity_diff <= 1.0
    assert 0.0 <= r.semantic_drift <= 1.0


@given(stores_strategy(), stores_strategy())
@settings(max_examples=40, deadline=None)
def test_invariant_identity_diff_symmetric(s1: MemoryStore, s2: MemoryStore) -> None:
    r12 = continuity_score(s1, s2)
    r21 = continuity_score(s2, s1)
    assert r12.identity_diff == pytest.approx(r21.identity_diff)


@given(stores_strategy(), stores_strategy())
@settings(max_examples=40, deadline=None)
def test_invariant_semantic_drift_symmetric(s1: MemoryStore, s2: MemoryStore) -> None:
    r12 = continuity_score(s1, s2)
    r21 = continuity_score(s2, s1)
    assert r12.semantic_drift == pytest.approx(r21.semantic_drift, abs=1e-9)


@given(stores_strategy(), stores_strategy())
@settings(max_examples=40, deadline=None)
def test_invariant_continuity_symmetric(s1: MemoryStore, s2: MemoryStore) -> None:
    r12 = continuity_score(s1, s2)
    r21 = continuity_score(s2, s1)
    assert r12.continuity == pytest.approx(r21.continuity)
