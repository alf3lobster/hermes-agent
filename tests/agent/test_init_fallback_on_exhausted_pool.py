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


class _TaggedPool:
    def __init__(self, provider, api_key, entry_id):
        self.provider = provider
        self.api_key = api_key
        self.entry_id = entry_id

    def has_credentials(self):
        return True

    def entry_id_for_api_key(self, api_key):
        return self.entry_id if api_key == self.api_key else None


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
        _is_azure_openai_url=lambda _url: False,
        _is_direct_openai_url=lambda _url: False,
        _provider_model_requires_responses_api=lambda _model, provider=None: False,
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


def test_init_fallback_becomes_ready_after_matching_pool_and_snapshot(monkeypatch):
    """Constructor fallback publishes readiness only after its route authority is complete."""
    from agent import agent_init

    primary_pool = _TaggedPool("openrouter", "primary-key", "primary-entry")
    fallback_pool = _TaggedPool("deepseek", "fallback-key", "fallback-entry")
    routed_fallback = _mock_client(
        api_key="fallback-key", base_url="https://api.deepseek.com/v1"
    )
    readiness_writes = []

    class ObservedAgent(AIAgent):
        @property
        def _route_client_ready(self):
            return self.__dict__.get("_route_client_ready_value", True)

        @_route_client_ready.setter
        def _route_client_ready(self, value):
            readiness_writes.append(
                (
                    value,
                    getattr(self, "provider", None),
                    getattr(self, "_credential_pool", None),
                    getattr(self, "_credential_pool_entry_id", None),
                    hasattr(self, "_primary_runtime"),
                )
            )
            self.__dict__["_route_client_ready_value"] = value

    def resolve(provider, **_kwargs):
        if provider == "openrouter":
            return None, None
        assert provider == "deepseek"
        return routed_fallback, "deepseek-chat"

    snapshot_readiness = []
    original_snapshot = agent_init._snapshot_primary_runtime

    def observe_snapshot(agent):
        snapshot_readiness.append(agent._route_client_ready)
        original_snapshot(agent)

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: fallback_pool)
    monkeypatch.setattr(agent_init, "_snapshot_primary_runtime", observe_snapshot)

    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = ObservedAgent(
            provider="openrouter",
            model="primary-model",
            credential_pool=primary_pool,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{
                "provider": "deepseek",
                "model": "deepseek-chat",
                "tool_policy": "none",
            }],
        )

    assert snapshot_readiness == [False]
    assert readiness_writes == [
        (False, "deepseek", primary_pool, None, False),
        (True, "deepseek", fallback_pool, "fallback-entry", True),
    ]
    assert agent._credential_pool is fallback_pool
    assert agent._credential_pool_entry_id == "fallback-entry"
    assert agent._primary_credential_pool is fallback_pool


def test_init_fallback_materializes_native_anthropic_before_readiness(monkeypatch):
    """Constructor fallback binds the final Anthropic wire before snapshot/publication."""
    from agent import agent_init

    primary_pool = _TaggedPool("openrouter", "primary-key", "primary-entry")
    fallback_pool = _TaggedPool("anthropic", "fallback-key", "fallback-entry")
    routed_fallback = _mock_client(
        api_key="fallback-key", base_url="https://api.anthropic.com"
    )
    native_client = object()
    snapshot_states = []
    original_snapshot = agent_init._snapshot_primary_runtime

    def resolve(provider, **_kwargs):
        if provider == "openrouter":
            return None, None
        assert provider == "anthropic"
        return routed_fallback, "claude-sonnet-4-6"

    def observe_snapshot(agent):
        snapshot_states.append(
            (
                agent._route_client_ready,
                agent.client,
                agent._anthropic_client,
                agent._credential_pool,
                agent._credential_pool_entry_id,
            )
        )
        original_snapshot(agent)

    openai_constructor = MagicMock(return_value=MagicMock())
    build_anthropic = MagicMock(return_value=native_client)
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: fallback_pool)
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", build_anthropic)
    monkeypatch.setattr(agent_init, "_snapshot_primary_runtime", observe_snapshot)

    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", openai_constructor),
    ):
        agent = AIAgent(
            provider="openrouter",
            model="primary-model",
            credential_pool=primary_pool,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "tool_policy": "none",
            }],
        )

    openai_constructor.assert_not_called()
    build_anthropic.assert_called_once_with(
        "fallback-key", "https://api.anthropic.com", timeout=None
    )
    assert snapshot_states == [
        (False, None, native_client, fallback_pool, "fallback-entry")
    ]
    assert agent.api_mode == "anthropic_messages"
    assert agent.api_key == agent._anthropic_api_key == "fallback-key"
    assert agent._anthropic_base_url == "https://api.anthropic.com"
    assert agent._is_anthropic_oauth is False
    assert agent._primary_runtime["anthropic_api_key"] == "fallback-key"
    assert agent._route_client_ready is True


@pytest.mark.parametrize("api_mode", [None, "anthropic_messages"])
def test_init_fallback_materializes_bedrock_wire_before_readiness(monkeypatch, api_mode):
    """Constructor fallback binds either native Bedrock wire without an OpenAI client."""
    from agent import agent_init

    primary_pool = _TaggedPool("openrouter", "primary-key", "primary-entry")
    fallback_pool = _TaggedPool("bedrock", "aws-sdk", "bedrock-entry")
    routed_fallback = _mock_client(
        api_key="aws-sdk",
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com",
    )
    bind_calls = []
    snapshot_states = []
    original_snapshot = agent_init._snapshot_primary_runtime

    def resolve(provider, **_kwargs):
        if provider == "openrouter":
            return None, None
        assert provider == "bedrock"
        return routed_fallback, "anthropic.claude-sonnet-4-6-v1:0"

    def bind_bedrock(agent, base_url, api_mode):
        bind_calls.append((base_url, api_mode, agent._route_client_ready))
        agent._bedrock_region = "us-west-2"
        agent._bedrock_guardrail_config = None
        agent.client = None
        agent._client_kwargs = {}
        agent.api_key = agent._anthropic_api_key = "aws-sdk"
        agent._anthropic_base_url = base_url
        agent._is_anthropic_oauth = False
        agent._anthropic_client = None

    def observe_snapshot(agent):
        snapshot_states.append(
            (
                agent._route_client_ready,
                agent.client,
                agent._credential_pool,
                agent._credential_pool_entry_id,
            )
        )
        original_snapshot(agent)

    openai_constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: fallback_pool)
    monkeypatch.setattr("agent.bedrock_adapter.bind_bedrock_runtime", bind_bedrock)
    monkeypatch.setattr(agent_init, "_snapshot_primary_runtime", observe_snapshot)

    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", openai_constructor),
    ):
        agent = AIAgent(
            provider="openrouter",
            model="primary-model",
            credential_pool=primary_pool,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{
                "provider": "bedrock",
                "model": "anthropic.claude-sonnet-4-6-v1:0",
                "tool_policy": "none",
                **({"api_mode": api_mode} if api_mode else {}),
            }],
        )

    openai_constructor.assert_not_called()
    expected_mode = api_mode or "bedrock_converse"
    assert bind_calls == [
        ("https://bedrock-runtime.us-west-2.amazonaws.com", expected_mode, False)
    ]
    assert snapshot_states == [(False, None, fallback_pool, "bedrock-entry")]
    assert agent.api_mode == expected_mode
    assert agent._bedrock_region == "us-west-2"
    assert agent._route_client_ready is True


def test_init_fallback_mode_detection_failure_stays_inert(monkeypatch):
    """A committed fallback cannot inherit the unreachable primary's API mode."""
    routed_fallback = _mock_client(base_url="https://fallback.invalid/v1")
    readiness_writes = []

    class ObservedAgent(AIAgent):
        @property
        def _route_client_ready(self):
            return self.__dict__.get("_route_client_ready_value", True)

        @_route_client_ready.setter
        def _route_client_ready(self, value):
            readiness_writes.append(value)
            self.__dict__["_route_client_ready_value"] = value

    def resolve(provider, **_kwargs):
        if provider == "nous":
            return None, None
        assert provider == "my-provider"
        return routed_fallback, "fallback-model"

    from agent.client_lifecycle import _swap_fallback_clients

    swap_clients = MagicMock(wraps=_swap_fallback_clients)
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
    monkeypatch.setattr(
        "agent.chat_completion_helpers._fallback_api_mode_resolved",
        MagicMock(side_effect=ValueError("mode detection failed")),
    )
    monkeypatch.setattr("agent.client_lifecycle._swap_fallback_clients", swap_clients)

    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        with pytest.raises(
            RuntimeError,
            match="post-commit init fallback api mode detection failed for my-provider",
        ):
            ObservedAgent(
                provider="nous",
                model="primary-model",
                api_mode="codex_responses",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[{
                    "provider": "my-provider",
                    "model": "fallback-model",
                    "tool_policy": "none",
                }],
            )

    assert readiness_writes == [False]
    swap_clients.assert_not_called()


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
