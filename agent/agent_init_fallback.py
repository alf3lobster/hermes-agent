"""Init-time fallback chain: entry normalization, chain setup, and the fallback's api_mode."""

from __future__ import annotations

from typing import Any, Dict, List


def _client_kwargs_from_routed(client, timeout) -> dict[str, Any]:
    """OpenAI kwargs preserving the provider headers baked into a routed client."""
    kwargs = {"api_key": client.api_key, "base_url": str(client.base_url)}
    if timeout is not None:
        kwargs["timeout"] = timeout
    headers = (
        getattr(client, "_custom_headers", None)
        or getattr(client, "default_headers", None)
        or getattr(client, "_default_headers", None)
    )
    if headers:
        kwargs["default_headers"] = dict(headers)
    return kwargs


def _fallback_entries(fallback_model) -> list[dict[str, Any]]:
    """Normalize legacy single-dict ``fallback_model`` / list ``fallback_providers``."""
    if isinstance(fallback_model, dict):
        fallback_model = [fallback_model]
    if not isinstance(fallback_model, list):
        return []
    return [
        f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]


def _init_fallback_chain(agent, fallback_model):
    if getattr(agent, "_fallback_activated", False):
        from agent.chat_completion_helpers import _rebind_fallback_credential_pool
        _rebind_fallback_credential_pool(agent, agent.provider, agent.model)

    # Stable pool-entry identity: OAuth refreshes can replace the token before a failed
    # request is recovered, so the key value alone can't attribute the failure.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    # Ordered backups tried when the primary is exhausted (legacy single-dict or list).
    agent._fallback_chain = _fallback_entries(fallback_model)
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    agent._provider_fallback_tool_policy = getattr(agent, "_provider_fallback_tool_policy", "inherit")
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    chain = agent._fallback_chain
    if chain and not agent.quiet_mode:
        labels = [f"{f['model']} ({f['provider']})" for f in chain]
        if len(chain) == 1:
            print(f"🔄 Fallback model: {labels[0]}")
        else:
            print(f"🔄 Fallback chain ({len(chain)} providers): " + " → ".join(labels))


def recompute_init_fallback_api_mode(agent, fb_client, entry: dict[str, Any] | None = None) -> None:
    """Give an init-time fallback its own api_mode instead of the unreachable primary's.

    A Nous primary configured with ``api_mode: codex_responses`` otherwise leaves a Copilot
    gpt-5-mini fallback on the Responses API, which silently drops reasoning content (#46527).
    The per-turn path (``try_activate_fallback``) already recomputes via
    ``_fallback_api_mode_resolved``; this keeps init in agreement, and runs before the
    ``_primary_runtime`` snapshot so ``restore_primary_runtime`` replays the recomputed mode.
    Detection failure is post-publication failure: the committed fallback stays inert rather
    than inheriting the unreachable primary route's API mode.
    """
    from agent.chat_completion_helpers import _fallback_api_mode_hint, _fallback_api_mode_resolved
    from hermes_cli.providers import is_actual_route
    try:
        client_base_url = str(getattr(fb_client, "base_url", "") or "")
        explicit, hinted_mode = _fallback_api_mode_hint(
            entry or {}, agent.provider, str((entry or {}).get("base_url") or "") or None,
        )
        agent.api_mode = (
            "chat_completions"
            if is_actual_route(agent.provider, client_base_url)
            else hinted_mode
            if explicit or hinted_mode != "chat_completions"
            else _fallback_api_mode_resolved(
                agent, agent.provider, agent.model, client_base_url,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"post-commit init fallback api mode detection failed for {agent.provider}"
        ) from exc
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
