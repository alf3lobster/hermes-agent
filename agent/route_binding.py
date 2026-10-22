"""Construct and publish provider routes without exposing partial runtime state.

``bind_route_entry`` stages fallback/voice metadata with a credential-inert client shell,
then publishes before credential/client materialisation; ``reinstall_primary_runtime``
applies the same publication rule when restoring the primary route.
"""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

from utils import base_url_host_matches
from agent.route_binding_state import rebind_primary_credential_pool

logger = logging.getLogger(__name__)

_DEFERRED_ROUTE_CREDENTIAL = "route-preparation-uncredentialed"
_CREDENTIAL_FREE_ROUTE_DEFAULTS = {
    # ``openai`` is the documented direct-API alias for ``openai-api``.
    "openai": "https://api.openai.com/v1",
    # A remote/custom Ollama route carries an explicit URL; only the bare local
    # alias falls back to the standard loopback endpoint.
    "ollama": "http://127.0.0.1:11434/v1",
}


def _prepared_route_base_url(entry: dict[str, Any], provider: str) -> Optional[str]:
    """Resolve endpoint metadata without opening any credential source."""
    explicit = str(entry.get("base_url") or "").strip().rstrip("/")
    if explicit:
        return explicit
    if provider == "moa":
        return "moa://local"
    if provider in _CREDENTIAL_FREE_ROUTE_DEFAULTS:
        return _CREDENTIAL_FREE_ROUTE_DEFAULTS[provider]
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
        profiled = str(getattr(profile, "base_url", "") or "").strip().rstrip("/")
        if profiled:
            return profiled
    except Exception:
        logger.debug("Provider profile unavailable during route preparation", exc_info=True)
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        config = PROVIDER_REGISTRY.get(provider)
        registered = str(getattr(config, "inference_base_url", "") or "").strip().rstrip("/")
        if registered:
            return registered
    except Exception:
        logger.debug("Provider registry metadata unavailable during route preparation", exc_info=True)
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider
        custom = _get_named_custom_provider(provider) or {}
        configured = str(custom.get("base_url") or "").strip().rstrip("/")
        if configured:
            return configured
    except Exception:
        logger.debug("Named-provider metadata unavailable during route preparation", exc_info=True)
    return None


def _prepared_route_client(entry: dict[str, Any], provider: str) -> Optional[Any]:
    """Credential-inert client shell used only until post-publication maintenance runs."""
    base_url = _prepared_route_base_url(entry, provider)
    if not base_url:
        return None
    return SimpleNamespace(base_url=base_url, api_key=_DEFERRED_ROUTE_CREDENTIAL)


def _install_prepared_route_client(agent: Any, client: Any, base_url: str, api_mode: str) -> None:
    """Install a deliberately unusable client shell without invoking provider auth builders."""
    agent._route_client_ready = False
    agent.api_key = _DEFERRED_ROUTE_CREDENTIAL
    agent._client_kwargs = {
        "api_key": _DEFERRED_ROUTE_CREDENTIAL,
        "base_url": base_url,
    }
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = ""
    agent._is_anthropic_oauth = False
    if api_mode == "anthropic_messages":
        agent.client = None
        agent._anthropic_client = client
        agent._anthropic_api_key = _DEFERRED_ROUTE_CREDENTIAL
        agent._anthropic_base_url = base_url
    else:
        agent.client = client


def _stage_route_agent(agent: Any) -> Any:
    """Copy the live route while detaching objects whose preparation has durable effects."""
    staged = copy.copy(agent)
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        staged.context_compressor = copy.copy(compressor)
        staged.context_compressor._session_db = None
        staged.context_compressor._session_id = ""
    if hasattr(agent, "_transport_cache"):
        staged._transport_cache = dict(agent._transport_cache)
    for name in ("request_overrides", "reasoning_config", "runtime_capabilities"):
        value = getattr(agent, name, None)
        if isinstance(value, dict):
            setattr(staged, name, copy.deepcopy(value))
    return staged


def _revalidate_committed_compression(agent: Any) -> None:
    """Refresh an existing feasibility verdict only after route publication."""
    if getattr(agent, "_compression_feasibility_checked", False) is True:
        from agent.conversation_compression import revalidate_compression_feasibility
        revalidate_compression_feasibility(agent)


def _commit_route_agent(agent: Any, staged: Any) -> None:
    """Publish one prepared route, then persist its already-decided compressor reset state."""
    live_compressor = getattr(agent, "context_compressor", None)
    staged_compressor = getattr(staged, "context_compressor", None)
    if staged_compressor is not None and live_compressor is not None:
        staged_compressor._session_db = getattr(live_compressor, "_session_db", None)
        staged_compressor._session_id = getattr(live_compressor, "_session_id", "")
    vars(agent).update(vars(staged))
    if staged_compressor is not None:
        from agent.route_binding_state import persist_prepared_compressor
        try:
            persist_prepared_compressor(live_compressor, staged_compressor)
        except Exception:
            logger.warning(
                "Route committed but compressor-state persistence failed",
                exc_info=True,
            )


def _bind_route_entry_in_place(
    agent: Any, entry: dict[str, Any], provider: str, model: str, *, owner: Any = None,
) -> Optional[tuple[str, str]]:
    """Swap ``agent`` onto ``entry`` (a fallback-chain-shaped dict). Returns ``(old_model,
    old_provider)``, or None when the provider has no usable client. Sets ``_fallback_activated``
    so the next turn's ``restore_primary_runtime`` reverts it; raises on a failed swap."""
    from agent.chat_completion_helpers import (
        _fallback_api_mode_hint, _fallback_api_mode_resolved, _rebind_fallback_credential_pool,
        _reresolve_fallback_reasoning_config, _rescope_fallback_extra_body, _update_fallback_context_compressor,
    )
    # Preparation uses endpoint metadata plus a non-secret dead credential. The real
    # provider resolver may heal/select persistent pools, so it runs only after publication.
    base_url_hint = (entry.get("base_url") or "").strip() or None
    api_mode_explicit, api_mode = _fallback_api_mode_hint(entry, provider, base_url_hint)
    client = _prepared_route_client(entry, provider)
    if client is None:
        return None
    if provider == "moa":
        # A MoA entry means the preset itself, exactly like ``provider: moa`` in config. The client
        # is the preset's aggregator and only proves the preset resolves; bind the facade with the
        # same pins every other MoA build site uses (#112525, #112623).
        base_url, api_mode = "moa://local", "chat_completions"
    else:
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider
            model = normalize_model_for_provider(model, provider)
        except Exception as norm_err:  # health: allow BLE001 -- moved verbatim from try_activate_fallback; a bad catalog entry keeps the raw id
            logger.warning("Could not normalize fallback model %r for provider %r: %s", model, provider, norm_err)
        base_url = str(client.base_url)
        from hermes_cli.providers import is_actual_route
        if is_actual_route(provider, base_url):
            api_mode = "chat_completions"
        elif not api_mode_explicit and api_mode == "chat_completions":
            api_mode = _fallback_api_mode_resolved(agent, provider, model, base_url)

    old_model, old_provider, old_base_url = agent.model, agent.provider, agent.base_url
    # Clear the per-config context_length override so the new model's own context window is
    # resolved instead of the previous model's stale value (#22387).
    agent._config_context_length = None
    agent.model, agent.provider, agent.requested_provider = model, provider, provider
    agent.base_url, agent.api_mode = base_url, api_mode
    # reasoning_content echo opt-in travels with the active provider; restore_primary_runtime reverts it.
    agent._reasoning_echo_flag = bool(entry.get("reasoning_echo", False))
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    from agent.turn_recovery_codex import reset_codex_reasoning_replay
    reset_codex_reasoning_replay(agent)
    agent._fallback_activated = True

    _rebind_fallback_credential_pool(agent, provider, model, load_if_missing=False)
    agent._route_client_ready = False
    if provider == "moa":
        from agent.moa_loop import bind_moa_runtime
        bind_moa_runtime(agent, model, owner=owner)
    else:
        _install_prepared_route_client(agent, client, base_url, api_mode)

    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
        provider=provider, base_url=base_url, api_mode=api_mode, model=model)
    _update_fallback_context_compressor(agent, allow_probe=False)
    _reresolve_fallback_reasoning_config(agent)
    _rescope_fallback_extra_body(agent, old_model, old_provider, old_base_url)
    return old_model, old_provider


def _materialize_route_client(agent: Any, entry: dict[str, Any], provider: str) -> None:
    """Install real credentials/client after the restrictive route is already live."""
    if provider == "moa":
        from agent.moa_loop import bind_moa_runtime
        bind_moa_runtime(agent, agent.model, owner=agent)
        agent._route_client_ready = True
        return  # The facade resolves its advisors/aggregator lazily on the committed route.
    from agent.auxiliary_client import resolve_provider_client
    from agent.chat_completion_helpers import (
        _fallback_api_mode_resolved, _rebind_fallback_credential_pool,
        _update_fallback_context_compressor,
    )
    from hermes_cli.fallback_config import resolve_entry_api_key

    base_url_hint = (entry.get("base_url") or "").strip() or None
    api_key_hint = resolve_entry_api_key(entry)
    if base_url_hint and base_url_host_matches(base_url_hint, "ollama.com") and not api_key_hint:
        from agent.secret_scope import get_secret
        api_key_hint = get_secret("OLLAMA_API_KEY") or None
    client, _resolved_model = resolve_provider_client(
        provider,
        model=agent.model,
        raw_codex=True,
        explicit_base_url=base_url_hint,
        explicit_api_key=api_key_hint,
        api_mode=agent.api_mode,
    )
    if client is None:
        raise RuntimeError(f"post-commit client resolution failed for {provider}")
    base_url = str(client.base_url)
    api_mode = agent.api_mode
    from hermes_cli.providers import is_actual_route
    if is_actual_route(provider, base_url):
        api_mode = "chat_completions"
    elif not entry.get("api_mode") and api_mode == "chat_completions":
        api_mode = _fallback_api_mode_resolved(agent, provider, agent.model, base_url)
    agent.base_url, agent.api_mode = base_url, api_mode
    from agent.client_lifecycle import _swap_fallback_clients
    _swap_fallback_clients(agent, client, provider, agent.model, base_url, api_mode)
    agent._route_client_ready = True
    _rebind_fallback_credential_pool(agent, provider, agent.model)
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)
    agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
        provider=provider, base_url=base_url, api_mode=api_mode, model=agent.model,
    )
    agent._ensure_lmstudio_runtime_loaded()
    _update_fallback_context_compressor(agent)


def bind_route_entry(
    agent: Any,
    entry: dict[str, Any],
    provider: str,
    model: str,
    *,
    admit: Optional[Callable[[Any], None]] = None,
) -> Optional[tuple[str, str]]:
    """Prepare a complete route without live effects, then publish it once."""
    from hermes_cli.fallback_config import fallback_tool_policy
    staged = _stage_route_agent(agent)
    staged._provider_fallback_tool_policy = fallback_tool_policy(entry)
    bound = _bind_route_entry_in_place(staged, entry, provider, model, owner=agent)
    if bound is not None:
        from agent.chat_completion_helpers import _reset_stale_streak, rewrite_prompt_model_identity
        from agent.native_compaction import resolve_native_compaction_capabilities

        rewrite_prompt_model_identity(staged, staged.model, provider)
        staged._provider_fallback_active = True
        staged._provider_fallback_route = (str(staged.model), str(provider))
        _reset_stale_streak(staged)
        staged.runtime_capabilities = resolve_native_compaction_capabilities(
            model=staged.model,
            base_url=staged.base_url,
            provider=provider,
            is_codex_backend=provider == "openai-codex",
        )
        if admit is not None:
            admit(staged)
        _commit_route_agent(agent, staged)
        try:
            _materialize_route_client(agent, entry, provider)
        except Exception:
            logger.warning(
                "Fallback route committed but credential/client maintenance failed for %s/%s",
                provider,
                agent.model,
                exc_info=True,
            )
    return bound


def _reinstall_runtime_snapshot_in_place(
    agent: Any, rt: dict[str, Any], *, defer_client: bool = False,
) -> None:
    """Put ``agent`` on the runtime recorded in ``rt`` (a ``_primary_runtime``-shaped snapshot):
    identity, client, caching flags, compressor, reasoning and prompt identity. Credential pool
    and fallback bookkeeping are the caller's. Raises on failure."""
    target_tool_policy = rt.get("fallback_tool_policy", "inherit")
    current_tool_policy = getattr(agent, "_provider_fallback_tool_policy", "inherit")
    # Until the route restore is complete, retain the stricter authority of the two routes.
    agent._provider_fallback_tool_policy = (
        "none" if "none" in (current_tool_policy, target_tool_policy) else "inherit"
    )
    from agent.agent_runtime_helpers import (
        _apply_primary_runtime_fields, _rebuild_primary_client, _restore_runtime_capabilities,
    )
    _apply_primary_runtime_fields(agent, rt)
    from agent.turn_recovery_codex import reset_codex_reasoning_replay
    reset_codex_reasoning_replay(agent)
    _restore_runtime_capabilities(agent, rt)
    agent._use_prompt_caching = rt["use_prompt_caching"]
    # Default to native layout for snapshots predating the native-vs-proxy split.
    agent._use_native_cache_layout = rt.get(
        "use_native_cache_layout",
        agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
    )
    # An operator cache disable (_cache_disabled) must survive snapshot restoration.
    if getattr(agent, "_cache_disabled", False):
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
    if defer_client:
        shell = SimpleNamespace(base_url=agent.base_url, api_key=_DEFERRED_ROUTE_CREDENTIAL)
        _install_prepared_route_client(agent, shell, agent.base_url, agent.api_mode)
    else:
        _rebuild_primary_client(
            agent, rt, reason="restore_primary", owner=getattr(agent, "_route_owner", None)
        )
        agent._route_client_ready = True
    agent.context_compressor.update_model(
        model=rt["compressor_model"], context_length=rt["compressor_context_length"],
        base_url=rt["compressor_base_url"], api_key=rt["compressor_api_key"],
        provider=rt["compressor_provider"], api_mode=rt.get("compressor_api_mode", ""),
    )
    # A staged restore cannot resolve auxiliary clients or probe providers before publication.
    if not defer_client and getattr(agent, "_compression_feasibility_checked", False) is True:
        from agent.conversation_compression import revalidate_compression_feasibility
        revalidate_compression_feasibility(agent)
    # Older snapshots have no reasoning_config; keep the current value.
    saved_reasoning = rt.get("reasoning_config")
    if saved_reasoning is not None:
        agent.reasoning_config = dict(saved_reasoning)
    # Reset the stale-call circuit breaker: its streak measured the route being left.
    from agent.chat_completion_helpers import _reset_stale_streak, rewrite_prompt_model_identity
    _reset_stale_streak(agent)
    # Undo the identity rewrite so the prompt is byte-identical to the stored copy again
    # (prefix cache match).
    rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])
    agent._provider_fallback_tool_policy = target_tool_policy


def reinstall_runtime_snapshot(agent: Any, rt: dict[str, Any]) -> None:
    """Prepare a restored route without live effects, then publish it once."""
    current_tool_policy = getattr(agent, "_provider_fallback_tool_policy", "inherit")
    target_tool_policy = rt.get("fallback_tool_policy", "inherit")
    staged = _stage_route_agent(agent)
    staged._route_owner = agent
    try:
        _reinstall_runtime_snapshot_in_place(staged, rt, defer_client=True)
    except Exception:
        agent._provider_fallback_tool_policy = (
            "none" if "none" in (current_tool_policy, target_tool_policy) else "inherit"
        )
        raise
    del staged._route_owner
    _commit_route_agent(agent, staged)
    try:
        from agent.agent_runtime_helpers import _rebuild_primary_client
        _rebuild_primary_client(agent, rt, reason="restore_snapshot", owner=agent)
        agent._route_client_ready = True
        _revalidate_committed_compression(agent)
    except Exception:
        logger.warning(
            "Runtime route committed but client maintenance failed for %s/%s",
            rt.get("provider", ""),
            rt.get("model", ""),
            exc_info=True,
        )


def reinstall_primary_runtime(
    agent: Any, rt: dict[str, Any], primary_provider: str, primary_model: str, matches_primary,
    load_primary_pool, prefetched_pool=None, prefetched: bool = False,
    *, probe_codex_reopen: bool = False,
) -> None:
    """Put ``agent`` back on its ``_primary_runtime`` snapshot ``rt`` and clear the fallback state."""
    current_tool_policy = getattr(agent, "_provider_fallback_tool_policy", "inherit")
    target_tool_policy = rt.get("fallback_tool_policy", "inherit")
    staged = _stage_route_agent(agent)
    staged._route_owner = agent
    try:
        _reinstall_runtime_snapshot_in_place(staged, rt, defer_client=True)
        staged._fallback_activated = False
        staged._fallback_index = 0
        staged._rate_limit_backoff_count = 0
        staged._provider_fallback_active = False
        staged._provider_fallback_route = None
    except Exception:
        agent._provider_fallback_tool_policy = (
            "none" if "none" in (current_tool_policy, target_tool_policy) else "inherit"
        )
        raise
    del staged._route_owner
    _commit_route_agent(agent, staged)
    try:
        if probe_codex_reopen:
            import time

            agent._codex_reopen_checked_at = time.monotonic()
            pool = prefetched_pool if prefetched else load_primary_pool()
            if pool is None or not matches_primary(pool) or not pool.lift_reopened_cooldowns(
                model=primary_model or None,
            ):
                logger.info(
                    "Primary Codex route committed but quota remains unavailable; client stays inert",
                )
                return
            agent._rate_limited_until = 0
            prefetched_pool, prefetched = pool, True
        from agent.agent_runtime_helpers import _rebuild_primary_client
        _rebuild_primary_client(agent, rt, reason="restore_primary", owner=agent)
        agent._route_client_ready = True
        _revalidate_committed_compression(agent)
        rebind_primary_credential_pool(
            agent, primary_provider, primary_model, matches_primary,
            load_primary_pool, prefetched_pool, prefetched,
        )
        agent._primary_credential_pool = getattr(agent, "_credential_pool", None)
    except Exception:
        logger.warning(
            "Primary route committed but client/credential maintenance failed for %s/%s",
            primary_provider,
            primary_model,
            exc_info=True,
        )
