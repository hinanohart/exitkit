# SPDX-License-Identifier: Apache-2.0
"""Cedar 4.8 audit policy loader for ExitKit.

Loads a Cedar policy set from a ``.cedar`` file and evaluates authz decisions
for PAMSnapshot / ContinuerScore resources against AuditedRole principals.

This is an *audit policy layer*, not a security boundary.  The caller is
responsible for enforcement; this module records the authz decision for
observability and compliance purposes.

Requires the optional ``cedarpy`` package (``pip install exitkit[cedar]``).
``cedarpy`` wraps the Cedar Rust SDK and provides the ``is_authorized``
function used here.

Supported actions:  ``"read"``, ``"write"``, ``"decay"``
Supported resource types:  ``"PAMSnapshot"``, ``"ContinuerScore"``
Supported principal types:  ``"AuditedRole"`` (roles: reader / writer / admin)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VALID_ACTIONS: frozenset[str] = frozenset({"read", "write", "decay"})
_VALID_RESOURCE_TYPES: frozenset[str] = frozenset({"PAMSnapshot", "ContinuerScore"})


class PolicyLoadError(Exception):
    """Raised when a Cedar policy file cannot be loaded or parsed."""


class UnknownActionError(ValueError):
    """Raised when an unsupported action is requested."""


class Decision:
    """Represents an authz decision from the Cedar policy engine."""

    __slots__ = ("action", "allowed", "diagnostics", "principal", "resource")

    def __init__(
        self,
        *,
        allowed: bool,
        action: str,
        principal: str,
        resource: str,
        diagnostics: list[str] | None = None,
    ) -> None:
        self.allowed = allowed
        self.action = action
        self.principal = principal
        self.resource = resource
        self.diagnostics: list[str] = diagnostics or []

    def __repr__(self) -> str:
        verdict = "Allow" if self.allowed else "Deny"
        return (
            f"Decision({verdict} principal={self.principal!r} "
            f"action={self.action!r} resource={self.resource!r})"
        )


def load_policy_set(path: str | Path) -> str:
    """Load a Cedar policy set from *path* and return its text content.

    Args:
        path: Filesystem path to a ``.cedar`` policy file.

    Returns:
        The raw Cedar policy text ready for use with ``cedarpy.is_authorized``.

    Raises:
        PolicyLoadError: If the file does not exist, is not readable, or
            cannot be validated by the Cedar engine.
    """
    try:
        import cedarpy
    except ImportError as exc:
        raise PolicyLoadError(
            "cedarpy is required to load Cedar policies. "
            "Install it with: pip install exitkit[cedar]"
        ) from exc

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise PolicyLoadError(f"Cedar policy file not found: {resolved}")
    if not resolved.is_file():
        raise PolicyLoadError(f"Cedar policy path is not a file: {resolved}")

    try:
        policy_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(f"Cannot read Cedar policy file {resolved}: {exc}") from exc

    # Validate syntax via cedarpy (no schema required for syntax check).
    # cedarpy.validate_policies requires a schema; use format round-trip for
    # syntax-only check.
    try:
        cedarpy.format_policies(policy_text)
    except Exception as exc:  # cedarpy raises RuntimeError on parse failure
        raise PolicyLoadError(f"Cedar policy syntax error in {resolved}: {exc}") from exc

    logger.debug("Loaded Cedar policy set from %s (%d chars)", resolved, len(policy_text))
    return policy_text


def validate(
    policy_set: str,
    *,
    action: str,
    principal: str,
    resource: str,
    entities: list[dict[str, Any]] | None = None,
) -> Decision:
    """Evaluate a Cedar authz request against *policy_set*.

    Args:
        policy_set: Cedar policy text returned by :func:`load_policy_set`.
        action: One of ``"read"``, ``"write"``, ``"decay"``.
        principal: Cedar entity UID string for the requesting principal,
            e.g. ``'AuditedRole::"reader"'``.
        resource: Cedar entity UID string for the target resource,
            e.g. ``'PAMSnapshot::"snap-001"'``.
        entities: Optional list of Cedar entity dicts to inject into the
            evaluation context (parents, attributes).  Defaults to ``[]``.

    Returns:
        :class:`Decision` with ``allowed`` set according to the policy.

    Raises:
        UnknownActionError: If *action* is not one of the supported values.
        PolicyLoadError: If ``cedarpy`` is not installed.
    """
    try:
        import cedarpy
    except ImportError as exc:
        raise PolicyLoadError(
            "cedarpy is required to evaluate Cedar policies. "
            "Install it with: pip install exitkit[cedar]"
        ) from exc

    if action not in _VALID_ACTIONS:
        raise UnknownActionError(
            f"Unknown action {action!r}. Supported actions: {sorted(_VALID_ACTIONS)}"
        )

    request: dict[str, Any] = {
        "principal": principal,
        "action": f'Action::"{action}"',
        "resource": resource,
        "context": {},
    }

    entity_list: list[dict[str, Any]] = entities if entities is not None else []

    result = cedarpy.is_authorized(
        request=request,
        policies=policy_set,
        entities=entity_list,
    )

    allowed = result.allowed
    diag_reasons: list[str] = []
    try:
        diag = result.diagnostics
        if diag:
            reasons = getattr(diag, "reasons", [])
            errors = getattr(diag, "errors", [])
            diag_reasons = [str(r) for r in reasons] + [str(e) for e in errors]
    except Exception:  # diagnostics access is best-effort
        pass

    decision = Decision(
        allowed=allowed,
        action=action,
        principal=principal,
        resource=resource,
        diagnostics=diag_reasons,
    )

    verdict = "Allow" if allowed else "Deny"
    logger.debug(
        "Cedar authz %s: principal=%r action=%r resource=%r",
        verdict,
        principal,
        action,
        resource,
    )
    return decision
