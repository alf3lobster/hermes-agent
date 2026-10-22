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
    prepared._persist_ineffective_compression_count()
    prepared._clear_durable_proactive_prune_rearm()
    if changed:
        prepared._persist_fallback_compression_streak()
        prepared._durable_write(
            "clear_compression_failure_cooldown", "compression failure cooldown clear"
        )
        prepared._persist_consecutive_overload_aborts()


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
    pool_provider = str(getattr(pool, "provider", "") or "").strip().lower()
    if pool is not None and pool_provider and not matches_primary(pool):
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        try:
            agent._credential_pool = prefetched_pool if prefetched else load_primary_pool()
        except Exception as exc:
            logger.warning(
                "Restore could not reload primary credential pool for %s: %s",
                primary_provider,
                exc,
                exc_info=True,
            )
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
