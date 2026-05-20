# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0rc1] — 2026-05-21

### Added

- **Cedar 4.8 audit policy layer** (`exitkit.cedar_loader`): `load_policy_set`,
  `validate`, `Decision`, `PolicyLoadError`, `UnknownActionError`.
  Governs `read`/`write`/`decay` actions on `PAMSnapshot` and `ContinuerScore`
  resources by `AuditedRole` principals.
- `cedar/policies.cedar`: canonical Cedar 4.8 policy file bundled with the
  package (included in both wheel and sdist).
- Optional dependency group `cedar` (`pip install exitkit[cedar]`) pulls in
  `cedarpy>=0.3`.
- 27 new tests in `tests/test_cedar_loader.py` covering allow/deny/unknown-action/
  import-error/OSError paths.
- `.github/workflows/release.yml`: Sigstore Cosign keyless attestation +
  PyPI trusted publish via OIDC on `v*` tag push.

### Changed
- Relicense from Apache-2.0 to MIT for portfolio alignment.

## [0.1.1] — 2026-05-18

### Fixed

- `continuity_score` now raises `ValueError` when a `MemoryStore`
  contains duplicate MemoryObject IDs. Previously the internal index
  silently kept the last occurrence, which could produce false-positive
  `mutated` results when two stores held the same multi-set in different
  orders.

### Added

- `Embedder` type alias (`Callable[[Iterable[str]], np.ndarray]`) is now
  exported from the top-level `exitkit` package so callers can type-annotate
  their custom embedders without reaching into `exitkit.continuer`.
- New Hypothesis property test (`test_mutated_set_matches_flipped_ids`)
  that drives the `mutated` branch directly with `(same id, flipped
  content)` pairs.
- Focused unit test `test_duplicate_memory_ids_raise` covering the new
  fail-fast behaviour.
- `CHANGELOG.md` is now part of the sdist alongside `README.md` and
  `LICENSE`.

### Documentation

- Document that `semantic_drift` is bounded by `[0, 0.5]` for non-empty
  stores when the default `hashing_embedder` is used, and only reaches
  `1.0` when one store is empty or an embedder with negative-cosine
  outputs (e.g. sentence-transformers) is supplied.
- Document the effective weight asymmetry that follows from the
  `identity_diff` and default `semantic_drift` ranges differing.
- Document the `\w+` tokenisation limitation (content that is only
  punctuation / whitespace / emoji collapses to zero tokens under the
  default embedder).

### Internal

- Simplify the cosine-to-drift conversion to `(1 - cosine) / 2` (numerically
  identical to the previous `1 - (cosine + 1) / 2`, but easier to read as a
  half-angle distance).

## [0.1.0] — 2026-05-17

Initial public release.

### Added

- `continuity_score(before, after, *, identity_weight=0.5, semantic_weight=0.5, embedder=None)`
  — Nozickian closest-continuer metric over two `portable_ai_memory.MemoryStore`
  snapshots.
- `DriftReport` frozen dataclass exposing `continuity`, `identity_diff`,
  `semantic_drift`, `added` / `removed` / `mutated` ID sets, `weights`,
  and `n_before` / `n_after`.
- `hashing_embedder(texts, *, n_features=1024)` — pure-numpy, deterministic
  (blake2b) bag-of-words embedder used as the default semantic backend.
- 17 tests: edge-case units, 5 Hypothesis property invariants
  (reflexivity, unit-interval bound, identity-diff symmetry, semantic-drift
  symmetry, continuity symmetry), and a toy 50/50 ROC-AUC ≥ 0.7 benchmark.
- CI on Python 3.11 / 3.12 / 3.13 (ruff, mypy `--strict`, pytest with
  coverage).

### Deferred to a future release

- Cedar-based export policy gate (`cedarpy` 4.8 verified pip-installable,
  not integrated).
- Sigstore Cosign keyless manifest signing.
- PyPI trusted publishing.

[0.1.1]: https://github.com/hinanohart/exitkit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hinanohart/exitkit/releases/tag/v0.1.0
