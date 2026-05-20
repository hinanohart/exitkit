# SPDX-License-Identifier: Apache-2.0
"""Tests for exitkit.cedar_loader — Cedar 4.8 audit policy layer."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from exitkit.cedar_loader import (
    Decision,
    PolicyLoadError,
    UnknownActionError,
    load_policy_set,
    validate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_POLICIES_CEDAR = Path(__file__).parent.parent / "cedar" / "policies.cedar"


@pytest.fixture(scope="module")
def policy_set() -> str:
    """Load the canonical Cedar policy set bundled with ExitKit."""
    return load_policy_set(_POLICIES_CEDAR)


@pytest.fixture()
def minimal_policy(tmp_path: Path) -> str:
    """A minimal inline Cedar policy for isolated unit tests."""
    cedar_file = tmp_path / "minimal.cedar"
    cedar_file.write_text(
        textwrap.dedent(
            """
            permit (
                principal in AuditedRole::"reader",
                action == Action::"read",
                resource is PAMSnapshot
            );
            """
        ),
        encoding="utf-8",
    )
    return load_policy_set(cedar_file)


# ---------------------------------------------------------------------------
# load_policy_set tests
# ---------------------------------------------------------------------------


class TestLoadPolicySet:
    def test_loads_bundled_policy(self) -> None:
        """load_policy_set returns non-empty text from the bundled .cedar file."""
        text = load_policy_set(_POLICIES_CEDAR)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_returns_policy_text_string(self) -> None:
        """Return value contains recognisable Cedar keywords."""
        text = load_policy_set(_POLICIES_CEDAR)
        assert "permit" in text

    def test_missing_file_raises_policy_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyLoadError, match="not found"):
            load_policy_set(tmp_path / "nonexistent.cedar")

    def test_invalid_syntax_raises_policy_load_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.cedar"
        bad.write_text("this is not cedar syntax !!!@@@", encoding="utf-8")
        with pytest.raises(PolicyLoadError, match="syntax error"):
            load_policy_set(bad)

    def test_tmp_policy_loaded(self, tmp_path: Path) -> None:
        """A user-supplied .cedar file at a custom path loads correctly."""
        cedar_file = tmp_path / "custom.cedar"
        cedar_file.write_text(
            "permit (principal, action, resource);",
            encoding="utf-8",
        )
        text = load_policy_set(cedar_file)
        assert "permit" in text


# ---------------------------------------------------------------------------
# validate — allow cases
# ---------------------------------------------------------------------------


class TestValidateAllowed:
    def test_reader_can_read_pam_snapshot(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="read",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"snap-001"',
        )
        assert dec.allowed is True

    def test_reader_can_read_continuer_score(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="read",
            principal='AuditedRole::"reader"',
            resource='ContinuerScore::"score-001"',
        )
        assert dec.allowed is True

    def test_writer_can_write_pam_snapshot(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="write",
            principal='AuditedRole::"writer"',
            resource='PAMSnapshot::"snap-002"',
        )
        assert dec.allowed is True

    def test_writer_can_read_pam_snapshot(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="read",
            principal='AuditedRole::"writer"',
            resource='PAMSnapshot::"snap-003"',
        )
        assert dec.allowed is True

    def test_admin_can_decay_pam_snapshot(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="decay",
            principal='AuditedRole::"admin"',
            resource='PAMSnapshot::"snap-old"',
        )
        assert dec.allowed is True

    def test_admin_can_decay_continuer_score(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="decay",
            principal='AuditedRole::"admin"',
            resource='ContinuerScore::"score-old"',
        )
        assert dec.allowed is True


# ---------------------------------------------------------------------------
# validate — deny cases
# ---------------------------------------------------------------------------


class TestValidateDenied:
    def test_reader_cannot_write_pam_snapshot(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="write",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"snap-001"',
        )
        assert dec.allowed is False

    def test_reader_cannot_decay(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="decay",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"snap-001"',
        )
        assert dec.allowed is False

    def test_writer_cannot_decay(self, policy_set: str) -> None:
        dec = validate(
            policy_set,
            action="decay",
            principal='AuditedRole::"writer"',
            resource='PAMSnapshot::"snap-001"',
        )
        assert dec.allowed is False

    def test_unknown_principal_is_denied(self, policy_set: str) -> None:
        """A principal not in any AuditedRole group is denied by default."""
        dec = validate(
            policy_set,
            action="read",
            principal='User::"anonymous"',
            resource='PAMSnapshot::"snap-001"',
        )
        assert dec.allowed is False


# ---------------------------------------------------------------------------
# validate — unknown action
# ---------------------------------------------------------------------------


class TestValidateUnknownAction:
    def test_unknown_action_raises(self, policy_set: str) -> None:
        with pytest.raises(UnknownActionError, match="Unknown action"):
            validate(
                policy_set,
                action="delete",
                principal='AuditedRole::"admin"',
                resource='PAMSnapshot::"snap-001"',
            )

    def test_empty_action_raises(self, policy_set: str) -> None:
        with pytest.raises(UnknownActionError):
            validate(
                policy_set,
                action="",
                principal='AuditedRole::"reader"',
                resource='PAMSnapshot::"snap-001"',
            )


# ---------------------------------------------------------------------------
# Decision repr
# ---------------------------------------------------------------------------


class TestDecisionRepr:
    def test_allow_repr(self) -> None:
        d = Decision(
            allowed=True,
            action="read",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"x"',
        )
        assert "Allow" in repr(d)

    def test_deny_repr(self) -> None:
        d = Decision(
            allowed=False,
            action="write",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"x"',
        )
        assert "Deny" in repr(d)

    def test_diagnostics_default_empty(self) -> None:
        d = Decision(
            allowed=True,
            action="read",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"x"',
        )
        assert d.diagnostics == []


# ---------------------------------------------------------------------------
# minimal_policy isolation
# ---------------------------------------------------------------------------


class TestMinimalPolicy:
    def test_minimal_allow(self, minimal_policy: str) -> None:
        dec = validate(
            minimal_policy,
            action="read",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"snap-x"',
        )
        assert dec.allowed is True

    def test_minimal_deny_writer(self, minimal_policy: str) -> None:
        """Writer is not in the minimal policy — should be denied."""
        dec = validate(
            minimal_policy,
            action="read",
            principal='AuditedRole::"writer"',
            resource='PAMSnapshot::"snap-x"',
        )
        assert dec.allowed is False


# ---------------------------------------------------------------------------
# ImportError / OSError coverage (cedarpy absent or file unreadable)
# ---------------------------------------------------------------------------


class TestImportErrorPaths:
    def test_load_policy_set_no_cedarpy(self, tmp_path: Path) -> None:
        """load_policy_set raises PolicyLoadError when cedarpy is not installed."""
        cedar_file = tmp_path / "test.cedar"
        cedar_file.write_text("permit (principal, action, resource);", encoding="utf-8")

        original = sys.modules.pop("cedarpy", None)
        try:
            with (
                patch.dict(sys.modules, {"cedarpy": None}),
                pytest.raises(PolicyLoadError, match="cedarpy is required"),
            ):
                load_policy_set(cedar_file)
        finally:
            if original is not None:
                sys.modules["cedarpy"] = original

    def test_validate_no_cedarpy(self) -> None:
        """validate raises PolicyLoadError when cedarpy is not installed."""
        original = sys.modules.pop("cedarpy", None)
        try:
            with (
                patch.dict(sys.modules, {"cedarpy": None}),
                pytest.raises(PolicyLoadError, match="cedarpy is required"),
            ):
                validate(
                    "permit (principal, action, resource);",
                    action="read",
                    principal='AuditedRole::"reader"',
                    resource='PAMSnapshot::"snap-001"',
                )
        finally:
            if original is not None:
                sys.modules["cedarpy"] = original

    def test_load_policy_set_path_is_directory(self, tmp_path: Path) -> None:
        """load_policy_set raises PolicyLoadError when path is a directory."""
        with pytest.raises(PolicyLoadError, match="not a file"):
            load_policy_set(tmp_path)

    def test_load_policy_set_os_error(self, tmp_path: Path) -> None:
        """load_policy_set raises PolicyLoadError on read OSError."""
        cedar_file = tmp_path / "unreadable.cedar"
        cedar_file.write_text("permit (principal, action, resource);", encoding="utf-8")

        with (
            patch("exitkit.cedar_loader.Path.read_text", side_effect=OSError("perm denied")),
            pytest.raises(PolicyLoadError, match="Cannot read Cedar policy file"),
        ):
            load_policy_set(cedar_file)


class TestDiagnosticsEdgeCases:
    def test_decision_with_diagnostics_list(self) -> None:
        """Decision stores diagnostics when provided."""
        d = Decision(
            allowed=False,
            action="write",
            principal='AuditedRole::"reader"',
            resource='PAMSnapshot::"x"',
            diagnostics=["policy-1 denied", "no matching permit"],
        )
        assert len(d.diagnostics) == 2
        assert "policy-1 denied" in d.diagnostics
