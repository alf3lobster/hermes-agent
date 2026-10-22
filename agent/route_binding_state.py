"""Deferred side effects for prepared provider-route transitions."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    pool = getattr(agent, "_credential_pool", None)
    if pool is None or not matches_primary(pool):
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        agent._credential_pool = prefetched_pool if prefetched else load_primary_pool()
    agent._credential_pool_entry_id = None
    pool = getattr(agent, "_credential_pool", None)
    entry = pool.select(model=primary_model or None) if pool is not None else None
    if entry is None or not (
        getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    ):
        return
    if matches_primary(entry):
        agent._swap_credential(entry)
        logger.info(
            "Restore re-selected pool entry %s (%s)",
            getattr(entry, "id", "?"),
            getattr(entry, "label", "?"),
        )
    else:
        logger.info(
            "Restore skipped pool entry %s (%s): provider %s does not match primary provider %s",
            getattr(entry, "id", "?"),
            getattr(entry, "label", "?"),
            str(getattr(entry, "provider", "") or "").strip().lower() or "?",
            primary_provider or "?",
        )
