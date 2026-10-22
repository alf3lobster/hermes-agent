"""Network-inert context metadata for transactional route preparation."""

from __future__ import annotations

from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_FALLBACK_CONTEXT,
    _config_override_context_length,
    _longest_key_match,
    _strip_openrouter_routing_variant,
    _strip_provider_prefix,
)


def get_model_context_length_static(
    model: str,
    base_url: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """Return a context estimate without credentials, persistence or network probes.

    The committed route immediately refreshes this estimate through the full metadata
    resolver. Preparation admits only explicit configuration, provider-profile metadata
    and the built-in catalog.
    """
    if isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length
    if (provider or "").strip().lower() != "moa":
        context = _config_override_context_length(model, base_url, provider, custom_providers)
        if context is not None:
            return context
    if not str(model or "").strip():
        return DEFAULT_FALLBACK_CONTEXT
    model = _strip_provider_prefix(model)
    model = _strip_openrouter_routing_variant(model, base_url=base_url, provider=provider)
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    context = profile.get_model_context_length(model) if profile else None
    if type(context) is int and context > 0:
        return context
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    return hit[1] if hit else DEFAULT_FALLBACK_CONTEXT
