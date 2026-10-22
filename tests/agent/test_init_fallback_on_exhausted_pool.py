"""Regression test for #17929: AIAgent.__init__ should try fallback_model
when primary provider credentials are exhausted."""
import pytest
from unittest.mock import patch, MagicMock
from run_agent import AIAgent


def _make_tool_defs():
    return [{"type": "function", "function": {"name": "web_search",
             "description": "search", "parameters": {"type": "object", "properties": {}}}}]


def _mock_client(api_key="fb-key-1234567890", base_url="https://fb.example.com/v1"):
    c = MagicMock()
    c.api_key = api_key
    c.base_url = base_url
    c._default_headers = None
    return c


def test_init_fallback_publishes_restriction_before_credentials():
    """Credential resolution observes the committed response-only fallback route."""
    from types import SimpleNamespace

    from agent.agent_init import _routed_client_kwargs

    agent = SimpleNamespace(
        provider="primary",
        requested_provider="primary",
        model="primary-model",
        api_mode="chat_completions",
        _provider_fallback_tool_policy="inherit",
    )
    fallback = {
        "provider": "custom",
        "model": "fallback-model",
        "base_url": "https://fallback.invalid/v1",
        "tool_policy": "none",
    }
    observations = []

    def resolve_key(_entry):
        observations.append(
            (agent.provider, agent.model, agent._provider_fallback_tool_policy, agent._route_client_ready)
        )
        return ""

    client = _mock_client(base_url="https://fallback.invalid/v1")
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        side_effect=[(None, None), (client, "fallback-model")],
    ), patch("hermes_cli.fallback_config.resolve_entry_api_key", side_effect=resolve_key):
        _routed_client_kwargs(agent, [fallback], None)

    assert observations == [("custom", "fallback-model", "none", False)]


def test_init_fallback_does_not_reselect_after_post_publish_failure():
    """A committed restrictive init route cannot hop to a second candidate on failure."""
    from types import SimpleNamespace

    from agent.agent_init import _routed_client_kwargs

    agent = SimpleNamespace(
        provider="primary",
        requested_provider="primary",
        model="primary-model",
        api_mode="chat_completions",
        _provider_fallback_tool_policy="inherit",
    )
    attempted = []

    def resolve(provider, **_kwargs):
        attempted.append(provider)
        if provider == "primary":
            return None, None
        raise RuntimeError("first fallback failed")

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve), pytest.raises(
        RuntimeError, match="first fallback failed"
    ):
        _routed_client_kwargs(
            agent,
            [
                {"provider": "first", "model": "one", "tool_policy": "none"},
                {"provider": "second", "model": "two", "tool_policy": "none"},
            ],
            None,
        )

    assert attempted == ["primary", "first"]
    assert (agent.provider, agent.model, agent._provider_fallback_tool_policy) == (
        "first", "one", "none"
    )
    assert agent._route_client_ready is False


def test_init_tries_fallback_when_primary_returns_none():
    """When resolve_provider_client returns None for primary but succeeds for
    a fallback entry, __init__ should NOT raise RuntimeError."""
    fb = _mock_client()

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "tencent-token-plan":
            return fb, "kimi2.5"
        return None, None  # primary exhausted

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{
                "provider": "tencent-token-plan", "model": "kimi2.5", "tool_policy": "none",
            }],
        )
        assert agent.provider == "tencent-token-plan"
        assert agent.model == "kimi2.5"
        assert agent._fallback_activated is True
        assert agent._provider_fallback_tool_policy == "none"
        assert agent._route_client_ready is True
        assert agent._primary_runtime["fallback_tool_policy"] == "none"


def test_init_raises_when_no_fallback_configured():
    """When primary returns None and no fallback is set, should raise."""
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match="no API key was found"):
            AIAgent(
                provider="alibaba-coding-plan",
                model="qwen3.6-plus",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=None,
            )


def test_init_openrouter_failure_does_not_reselect_custom_fallback():
    """After publishing the first fallback, credential failure cannot hop to candidate two."""
    attempted = []

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        attempted.append(provider)
        return None, None

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        with pytest.raises(
            RuntimeError, match="post-commit init fallback client resolution failed for openrouter"
        ):
            AIAgent(
                provider="openrouter",
                model="z-ai/glm-5.2:free",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[
                    {"provider": "openrouter", "model": "poolside/laguna-s-2.1:free"},
                    {"provider": "custom", "model": "qwen3.5:4b",
                     "base_url": "http://127.0.0.1:11434/v1"},
                ],
            )

    assert attempted == ["openrouter", "openrouter"]


def test_init_single_fallback_failure_reports_committed_route():
    """A one-entry chain reports post-publication failure rather than generic setup advice."""
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        with pytest.raises(
            RuntimeError, match="post-commit init fallback client resolution failed for openrouter"
        ):
            AIAgent(
                provider="openrouter",
                model="z-ai/glm-5.2:free",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[{"provider": "openrouter",
                                 "model": "poolside/laguna-s-2.1:free"}],
            )


def test_init_fallback_recomputes_api_mode_for_copilot_gpt5_mini():
    """#46527: Nous primary (api_mode=codex_responses, no creds) -> Copilot gpt-5-mini
    fallback must end up with api_mode=chat_completions, not codex_responses.

    Copilot's gpt-5-mini is the documented exception that uses Chat Completions,
    not the Responses API. Before the fix, ``agent.api_mode`` stayed
    ``"codex_responses"`` after the init-time fallback activated, so every request
    for the fallback model went out via the Responses API and silently returned no
    reasoning/thinking content.
    """
    fb_client = _mock_client(base_url="https://api.githubcopilot.com")

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "copilot":
            return fb_client, "gpt-5-mini"
        return None, None  # primary (nous) has no usable credentials

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="nous",
            model="gpt-5.4-mini",
            api_mode="codex_responses",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "copilot", "model": "gpt-5-mini"}],
        )

        assert agent.provider == "copilot"
        assert agent.model == "gpt-5-mini"
        assert agent._fallback_activated is True
        assert agent.api_mode == "chat_completions"


def test_init_fallback_keeps_codex_responses_for_openai_codex():
    """#46527: Nous primary (api_mode=codex_responses, no creds) -> openai-codex
    fallback must keep api_mode=codex_responses (openai-codex always uses the
    Responses API)."""
    fb_client = _mock_client(base_url="https://chatgpt.com/backend-api/codex")

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "openai-codex":
            return fb_client, "gpt-5.4-mini"
        return None, None

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="nous",
            model="gpt-5.4-mini",
            api_mode="codex_responses",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "openai-codex", "model": "gpt-5.4-mini"}],
        )

        assert agent.provider == "openai-codex"
        assert agent.model == "gpt-5.4-mini"
        assert agent._fallback_activated is True
        assert agent.api_mode == "codex_responses"
