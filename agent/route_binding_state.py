"""Deferred side effects for prepared provider-route transitions."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def require_route_client_ready(agent: Any) -> None:
    """Refuse provider/client paths while a published route is incomplete."""
    readiness = getattr(agent, "__dict__", {}).get("_route_client_ready", True)
    if readiness is not True:
        raise RuntimeError("route client is not ready")


def persist_prepared_compressor(previous: Any, prepared: Any) -> None:
    """Persist reset values only after the prepared compressor is live."""
    changed = (
        getattr(previous, "model", None),
        getattr(previous, "provider", None),
        getattr(previous, "base_url", None),
        getattr(previous, "api_mode", None),
    ) != (
        getattr(prepared, "model", None),
        getattr(prepared, "provider", None),
        getattr(prepared, "base_url", None),
        getattr(prepared, "api_mode", None),
    )
    # An unbound compressor has no durable session state to maintain.
    session_id = getattr(prepared, "_session_id", "")
    if (
        not isinstance(session_id, str)
        or not session_id
        or getattr(prepared, "_session_db", None) is None
    ):
        return
    results = [
        ("ineffective compression count", prepared._persist_ineffective_compression_count()),
        ("proactive prune runway", prepared._clear_durable_proactive_prune_rearm()),
    ]
    if changed:
        results.extend([
            ("fallback compression streak", prepared._persist_fallback_compression_streak()),
            (
                "compression failure cooldown",
                prepared._durable_write(
                    "clear_compression_failure_cooldown",
                    "compression failure cooldown clear",
                ),
            ),
            ("compression overload streak", prepared._persist_consecutive_overload_aborts()),
        ])
    failed = [label for label, succeeded in results if succeeded is not True]
    if failed:
        raise RuntimeError("compressor persistence failed: " + ", ".join(failed))


def rebind_primary_credential_pool(
    agent: Any,
    primary_provider: str,
    primary_model: str,
    matches_primary: Any,
    load_primary_pool: Any,
    prefetched_pool: Any,
    prefetched: bool,
) -> None:
    """Select primary credentials only after the complete primary route is already live."""
    # This helper is used only for legacy snapshots, which carry no frozen pool
    # authority.  Never trust the currently attached pool merely because its provider
    # matches: a same-provider fallback may own a different account/entry ID.
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    pool = prefetched_pool if prefetched else load_primary_pool()
    if pool is None or not matches_primary(pool):
        return
    agent._credential_pool = pool
    entry = pool.select(model=primary_model or None) if pool is not None else None
    if entry is None:
        raise RuntimeError("primary credential pool has no selectable credential")
    if not (
        getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    ):
        raise RuntimeError("primary credential pool selected an unusable credential")
    if not matches_primary(entry):
        raise RuntimeError("primary credential pool selected a mismatched credential")
    if agent._swap_credential(entry) is not True:
        raise RuntimeError("credential-pool rebinding failed")
    logger.info(
        "Restore re-selected pool entry %s (%s)",
        getattr(entry, "id", "?"),
        getattr(entry, "label", "?"),
    )
