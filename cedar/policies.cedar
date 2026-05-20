// SPDX-License-Identifier: Apache-2.0
// ExitKit Cedar 4.8 audit policy layer.
// Governs read/write/decay actions on PAMSnapshot and ContinuerScore resources
// by AuditedRole principals.
//
// This is an audit policy layer, not a security boundary. It records authz
// decisions for observability and compliance; enforcement is the caller's
// responsibility.

// AuditedRole principals may read any PAMSnapshot.
permit (
    principal in AuditedRole::"reader",
    action == Action::"read",
    resource is PAMSnapshot
);

// AuditedRole writers may read or write PAMSnapshot resources.
permit (
    principal in AuditedRole::"writer",
    action in [Action::"read", Action::"write"],
    resource is PAMSnapshot
);

// AuditedRole principals may read ContinuerScore resources.
permit (
    principal in AuditedRole::"reader",
    action == Action::"read",
    resource is ContinuerScore
);

// AuditedRole writers may read ContinuerScore resources.
permit (
    principal in AuditedRole::"writer",
    action == Action::"read",
    resource is ContinuerScore
);

// AuditedRole admins may perform decay (retirement/expiry) on PAMSnapshot
// and ContinuerScore resources.
permit (
    principal in AuditedRole::"admin",
    action == Action::"decay",
    resource is PAMSnapshot
);

permit (
    principal in AuditedRole::"admin",
    action == Action::"decay",
    resource is ContinuerScore
);

// AuditedRole admins inherit all writer permissions.
permit (
    principal in AuditedRole::"admin",
    action in [Action::"read", Action::"write"],
    resource is PAMSnapshot
);

permit (
    principal in AuditedRole::"admin",
    action in [Action::"read", Action::"write"],
    resource is ContinuerScore
);
