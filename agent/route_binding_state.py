"""Pure preparation helpers for provider-route transactions."""

from __future__ import annotations

import copy
import logging
import random
import threading
import time
from dataclasses import replace
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pool_reservation_state(pool: Any) -> tuple[Any, ...]:
    entries = tuple(
        (
            entry.id,
            entry.priority,
            entry.request_count,
            entry.last_status,
            entry.last_status_at,
            entry.last_error_reset_at,
            tuple(sorted((entry.model_cooldowns or {}).items())),
            entry.runtime_api_key,
            entry.runtime_base_url,
        )
        for entry in pool._entries
    )
    return pool._current_id, pool._strategy, entries


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


def preview_credential(pool: Any, *, model: Optional[str] = None) -> Any:
    """Choose a usable credential without refresh, persistence, counters, or pool mutation."""
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        AUTH_TYPE_OAUTH,
        STATUS_DEAD,
        STATUS_EXHAUSTED,
        STRATEGY_LEAST_USED,
        STRATEGY_RANDOM,
        _exhausted_until,
        model_cooldown_until,
    )

    with pool._lock:
        now = time.time()
        sole_credential = pool._is_sole_credential()
        available = []
        for entry in pool._entries:
            if entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key:
                continue
            if entry.last_status == STATUS_DEAD:
                continue
            model_until = model_cooldown_until(entry, model)
            if model_until is not None and model_until > now:
                continue
            if entry.last_status == STATUS_EXHAUSTED:
                exhausted_until = _exhausted_until(entry, sole_credential=sole_credential)
                if exhausted_until is None or exhausted_until > now:
                    continue
            if entry.auth_type == AUTH_TYPE_OAUTH and not (entry.access_token or "").strip():
                continue
            available.append(entry)
        if not available:
            return None
        if pool._strategy == STRATEGY_RANDOM:
            return random.choice(available)
        if pool._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            return min(available, key=lambda entry: entry.request_count)
        return available[0]


def prepare_credential_pool(pool: Any, *, model: Optional[str] = None) -> tuple[Any, Any]:
    """Clone a pool and reserve one entry without mutating or persisting the source pool."""
    from agent.credential_pool import CredentialPool, STRATEGY_ROUND_ROBIN

    if not isinstance(pool, CredentialPool):
        # Preserve legacy custom/test pool identity. Transactional reservation is enforced for
        # the concrete CredentialPool used by production routing; foreign pool types retain their
        # existing selection contract.
        available = getattr(pool, "has_available", lambda **_kwargs: True)(model=model)
        entry = pool.select(model=model) if available else None
        return pool, entry

    with pool._lock:
        source_snapshot = _pool_reservation_state(pool)
        prepared = copy.copy(pool)
        for name, value in vars(pool).items():
            if isinstance(value, dict):
                setattr(prepared, name, dict(value))
            elif isinstance(value, list):
                setattr(prepared, name, list(value))
            elif isinstance(value, set):
                setattr(prepared, name, set(value))
        prepared._lock = threading.RLock()
        prepared._route_reservation_source = pool
        prepared._route_reservation_snapshot = source_snapshot

    entry = preview_credential(prepared, model=model)
    if entry is None:
        prepared._current_id = None
        return prepared, None
    entry = prepared._adopt(entry, persist=False, request_count=entry.request_count + 1)
    available = [candidate for candidate in prepared._entries if candidate.id != entry.id]
    if prepared._strategy == STRATEGY_ROUND_ROBIN and available:
        available.append(replace(entry, priority=len(prepared._entries) - 1))
        prepared._entries = [
            replace(candidate, priority=index) for index, candidate in enumerate(available)
        ]
        entry = prepared._find(lambda candidate: candidate.id == entry.id) or entry
    prepared._current_id = entry.id
    return prepared, entry


def validate_prepared_credential_pool(reservation: tuple[Any, Any]) -> None:
    """Reject a reservation when its source pool changed after the immutable snapshot."""
    from agent.credential_pool import CredentialPool

    prepared, _entry = reservation
    if not isinstance(prepared, CredentialPool):
        return
    source = prepared._route_reservation_source
    expected = prepared._route_reservation_snapshot
    with source._lock:
        current = _pool_reservation_state(source)
    if current != expected:
        raise RuntimeError("credential pool changed during route preparation")


def rebind_primary_credential_pool(
    agent: Any,
    primary_provider: str,
    primary_model: str,
    matches_primary: Any,
    load_primary_pool: Any,
    prefetched_pool: Any,
    prefetched: bool,
) -> None:
    """Prepare a primary pool and credential reservation without touching the source pool."""
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
    if pool is not None:
        reservation = prepare_credential_pool(pool, model=primary_model or None)
        agent._credential_pool, entry = reservation
        from agent.credential_pool import CredentialPool

        if isinstance(agent._credential_pool, CredentialPool):
            agent._route_credential_reservation = reservation
    else:
        entry = None
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
