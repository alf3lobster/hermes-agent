"""Fail-closed dispatch tests for response-only fallback routes."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _agent() -> AIAgent:
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "web_search tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def test_fallback_none_policy_blocks_every_emitted_call_before_dispatch():
    agent = _agent()
    agent._provider_fallback_tool_policy = "none"
    tool_calls = [
        _tool_call("read_file", '{"path":"customer-list.txt"}', "c1"),
        _tool_call("terminal", '{"command":"send-data"}', "c2"),
    ]
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    messages = []

    with (
        patch.object(agent, "_execute_tool_calls_sequential") as sequential,
        patch.object(agent, "_execute_tool_calls_concurrent") as concurrent,
        patch("model_tools.handle_function_call") as dispatch,
    ):
        agent._execute_tool_calls(assistant_message, messages, "task-1")

    sequential.assert_not_called()
    concurrent.assert_not_called()
    dispatch.assert_not_called()
    assert [message["tool_call_id"] for message in messages] == ["c1", "c2"]
    assert all(message["effect_disposition"] == "none" for message in messages)
    assert all("tool_policy=none" in message["content"] for message in messages)


def test_partial_runtime_fallback_activation_fails_closed():
    """A binder exception after route mutation must not leave a restricted route with tools."""
    from agent.chat_completion_helpers import try_activate_fallback

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "custom", "model": "restricted", "tool_policy": "none"}],
        _fallback_index=0,
        _unavailable_fallback_keys=set(),
        model="primary",
        provider="custom",
        base_url="https://primary.invalid/v1",
        _provider_fallback_tool_policy="inherit",
    )

    def partial_bind(target, _entry, _provider, _model):
        target.model = "restricted"
        target.base_url = "https://restricted.invalid/v1"
        raise RuntimeError("injected late bind failure")

    with (
        patch("agent.fallback_cooldown.switch_deferred_by_reset", return_value=False),
        patch("agent.fallback_cooldown._arm_rate_limit_cooldown", return_value=None),
        patch("agent.chat_completion_helpers._should_skip_fallback_candidate", return_value=False),
        patch("agent.chat_completion_helpers._fallback_chain_exhausted", return_value=False),
        patch("agent.route_binding._bind_route_entry_in_place", side_effect=partial_bind),
    ):
        assert try_activate_fallback(agent) is False

    assert agent.model == "primary"
    assert agent.base_url == "https://primary.invalid/v1"
    assert agent._provider_fallback_tool_policy == "inherit"


def test_failed_fallback_capability_resolution_does_not_publish_route(monkeypatch):
    """All route metadata must resolve before the prepared fallback becomes live."""
    from agent import route_binding

    agent = SimpleNamespace(
        model="primary",
        provider="custom",
        base_url="https://primary.invalid/v1",
        _provider_fallback_active=False,
        _provider_fallback_route=None,
        _provider_fallback_tool_policy="inherit",
        context_compressor=None,
    )

    def prepare_route(staged, *_args, **_kwargs):
        staged.model = "restricted"
        staged.provider = "custom"
        staged.base_url = "https://restricted.invalid/v1"
        return "primary", "custom"

    monkeypatch.setattr(route_binding, "_bind_route_entry_in_place", prepare_route)
    monkeypatch.setattr(
        "agent.native_compaction.resolve_native_compaction_capabilities",
        MagicMock(side_effect=RuntimeError("injected capability failure")),
    )

    with pytest.raises(RuntimeError, match="capability failure"):
        route_binding.bind_route_entry(
            agent,
            {"provider": "custom", "model": "restricted", "tool_policy": "none"},
            "custom",
            "restricted",
        )

    assert agent.model == "primary"
    assert agent.base_url == "https://primary.invalid/v1"
    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None
    assert agent._provider_fallback_tool_policy == "inherit"


def test_rejected_fallback_real_resolver_does_not_load_or_select_credential_pool(monkeypatch):
    """The real provider resolver must remain credential-inert until route publication."""
    from agent import auxiliary_client, route_binding

    agent = _agent()
    agent._credential_pool = None
    load_pool = MagicMock(side_effect=AssertionError("route preparation loaded credentials"))
    select_pool = MagicMock(side_effect=AssertionError("route preparation selected credentials"))
    fake_client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        api_key="route-preparation-placeholder",
    )

    monkeypatch.setattr("agent.credential_pool.load_pool", load_pool)
    monkeypatch.setattr(auxiliary_client, "_select_pool_entry", select_pool)
    monkeypatch.setattr(auxiliary_client, "_create_openai_client", lambda **_kwargs: fake_client)
    monkeypatch.setattr("agent.client_lifecycle._swap_fallback_clients", lambda *_args: None)
    monkeypatch.setattr("agent.agent_runtime_helpers.sync_credential_pool_entry_id", lambda *_args: None)
    monkeypatch.setattr(
        "agent.chat_completion_helpers._update_fallback_context_compressor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("agent.chat_completion_helpers._reresolve_fallback_reasoning_config", lambda *_args: None)
    monkeypatch.setattr("agent.chat_completion_helpers._rescope_fallback_extra_body", lambda *_args: None)
    monkeypatch.setattr(agent, "_ensure_lmstudio_runtime_loaded", lambda: None)
    monkeypatch.setattr(
        "agent.native_compaction.resolve_native_compaction_capabilities",
        MagicMock(side_effect=RuntimeError("injected capability failure")),
    )

    with pytest.raises(RuntimeError, match="capability failure"):
        route_binding.bind_route_entry(
            agent,
            {
                "provider": "openrouter",
                "model": "example/restricted:free",
                "tool_policy": "none",
            },
            "openrouter",
            "example/restricted:free",
        )

    load_pool.assert_not_called()
    select_pool.assert_not_called()


def test_candidate_screening_does_not_touch_attached_credential_pool(monkeypatch):
    """Candidate admission defers even retained-pool inspection until after publication."""
    from agent.chat_completion_helpers import _candidate_pool_exhausted

    pool = SimpleNamespace(
        provider="custom",
        has_credentials=MagicMock(side_effect=AssertionError("candidate screening inspected credentials")),
        has_available=MagicMock(side_effect=AssertionError("candidate screening healed credentials")),
        next_available_at=MagicMock(side_effect=AssertionError("candidate screening inspected cooldowns")),
    )
    agent = SimpleNamespace(_credential_pool=pool)
    load_pool = MagicMock(side_effect=AssertionError("candidate screening loaded credentials"))
    monkeypatch.setattr("agent.credential_pool.load_pool", load_pool)

    assert _candidate_pool_exhausted(agent, "custom", "candidate") is False
    pool.has_credentials.assert_not_called()
    pool.has_available.assert_not_called()
    pool.next_available_at.assert_not_called()
    load_pool.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("ollama", "http://127.0.0.1:11434/v1"),
    ],
)
def test_prepared_builtin_route_metadata_needs_no_credentials(provider, expected):
    """Common aliases remain useful without invoking their credential-aware resolver."""
    from agent.route_binding import _prepared_route_base_url

    assert _prepared_route_base_url({"provider": provider}, provider) == expected


def _fallback_restore_agent(primary_provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        _fallback_activated=True,
        _fallback_index=0,
        _primary_runtime={
            "provider": primary_provider,
            "model": "primary-model",
            "base_url": "https://primary.invalid/v1",
            "requested_provider": primary_provider,
            "fallback_tool_policy": "inherit",
        },
        _credential_pool=SimpleNamespace(provider="fallback-provider"),
        _rate_limited_until=0,
        _codex_reopen_checked_at=float("-inf"),
        _restore_wait_logged=False,
        _provider_fallback_active=True,
        _provider_fallback_route=("fallback-model", "fallback-provider"),
        _provider_fallback_tool_policy="none",
        provider="fallback-provider",
        model="fallback-model",
    )


def test_rejected_primary_restore_does_not_load_pool_before_route_preparation(monkeypatch):
    """The complete restoration entry point must not open a mismatched primary pool pre-commit."""
    from agent.agent_runtime_helpers import restore_primary_runtime

    agent = _fallback_restore_agent("custom")
    load_runtime_pool = MagicMock(side_effect=AssertionError("primary pool loaded before commit"))
    monkeypatch.setattr("agent.custom_entry_pool.load_runtime_pool", load_runtime_pool)
    monkeypatch.setattr("agent.fallback_cooldown._is_entitlement_rejected", lambda *_args: False)
    monkeypatch.setattr("hermes_cli.chat_catalog.is_known_non_chat_model", lambda *_args: False)
    monkeypatch.setattr(
        "agent.route_binding.reinstall_primary_runtime",
        MagicMock(side_effect=RuntimeError("primary preparation rejected")),
    )

    assert restore_primary_runtime(agent) is False
    load_runtime_pool.assert_not_called()
    assert agent._provider_fallback_tool_policy == "none"


def test_codex_early_reopen_does_not_load_primary_pool_before_commit(monkeypatch):
    """An early-reopen probe may inspect a retained primary pool, but must not load one."""
    from agent.agent_runtime_helpers import restore_primary_runtime

    agent = _fallback_restore_agent("openai-codex")
    agent._rate_limited_until = time.monotonic() + 3600
    load_runtime_pool = MagicMock(return_value=None)
    monkeypatch.setattr("agent.custom_entry_pool.load_runtime_pool", load_runtime_pool)
    monkeypatch.setattr("agent.fallback_cooldown._is_entitlement_rejected", lambda *_args: False)
    monkeypatch.setattr("hermes_cli.chat_catalog.is_known_non_chat_model", lambda *_args: False)

    assert restore_primary_runtime(agent) is False
    load_runtime_pool.assert_not_called()
    assert agent._provider_fallback_tool_policy == "none"


def test_post_publication_notice_failure_keeps_committed_fallback(monkeypatch):
    """Reporting failure cannot turn one committed fallback into another route attempt."""
    from agent.chat_completion_helpers import try_activate_fallback

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "custom", "model": "restricted", "tool_policy": "none"}],
        _fallback_index=0,
        _unavailable_fallback_keys=set(),
        model="primary",
        provider="custom",
        base_url="https://primary.invalid/v1",
        _provider_fallback_active=False,
        _provider_fallback_route=None,
        _provider_fallback_tool_policy="inherit",
    )

    def publish(target, _entry, provider, model):
        old = target.model, target.provider
        target.model, target.provider = model, provider
        target.base_url = "https://restricted.invalid/v1"
        target._provider_fallback_active = True
        target._provider_fallback_route = (model, provider)
        target._provider_fallback_tool_policy = "none"
        return old

    with (
        patch("agent.fallback_cooldown.switch_deferred_by_reset", return_value=False),
        patch("agent.fallback_cooldown._arm_rate_limit_cooldown", return_value=None),
        patch("agent.chat_completion_helpers._should_skip_fallback_candidate", return_value=False),
        patch("agent.route_binding.bind_route_entry", side_effect=publish),
        patch(
            "agent.chat_completion_helpers._buffer_fallback_notice",
            side_effect=RuntimeError("injected notice failure"),
        ),
    ):
        assert try_activate_fallback(agent) is True

    assert agent.model == "restricted"
    assert agent._provider_fallback_active is True
    assert agent._provider_fallback_route == ("restricted", "custom")
    assert agent._provider_fallback_tool_policy == "none"


def test_failed_same_provider_candidate_does_not_masquerade_as_committed(monkeypatch):
    """An older active route on the same provider cannot conceal a failed candidate bind."""
    from agent.chat_completion_helpers import try_activate_fallback

    agent = SimpleNamespace(
        _fallback_chain=[
            {"provider": "custom", "model": "rejected", "tool_policy": "none"},
            {"provider": "custom", "model": "accepted", "tool_policy": "none"},
        ],
        _fallback_index=0,
        _unavailable_fallback_keys=set(),
        model="old-fallback",
        provider="custom",
        base_url="https://old.invalid/v1",
        _provider_fallback_active=True,
        _provider_fallback_route=("old-fallback", "custom"),
        _provider_fallback_tool_policy="none",
        _rate_limited_until=0,
    )
    attempted = []

    def bind(target, _entry, provider, model):
        attempted.append(model)
        if model == "rejected":
            raise RuntimeError("candidate construction failed")
        old = (target.model, target.provider)
        target.model = model
        target.provider = provider
        target._provider_fallback_active = True
        target._provider_fallback_route = (model, provider)
        return old

    monkeypatch.setattr("agent.fallback_cooldown.switch_deferred_by_reset", lambda *_args: False)
    monkeypatch.setattr("agent.fallback_cooldown._arm_rate_limit_cooldown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent.chat_completion_helpers._should_skip_fallback_candidate", lambda *_args: False)
    monkeypatch.setattr("agent.route_binding.bind_route_entry", bind)
    monkeypatch.setattr("agent.chat_completion_helpers._buffer_fallback_notice", lambda *_args: None)
    monkeypatch.setattr("agent.chat_completion_helpers._log_fallback_activated", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_events.record_fallback", lambda **_kwargs: None)

    assert try_activate_fallback(agent) is True
    assert attempted == ["rejected", "accepted"]
    assert agent.model == "accepted"
    assert agent._provider_fallback_route == ("accepted", "custom")


def test_later_failed_candidate_cannot_widen_a_partially_bound_restricted_route():
    from agent.chat_completion_helpers import try_activate_fallback

    agent = SimpleNamespace(
        _fallback_chain=[
            {"provider": "custom", "model": "restricted", "tool_policy": "none"},
            {"provider": "custom", "model": "next", "tool_policy": "inherit"},
        ],
        _fallback_index=0,
        _unavailable_fallback_keys=set(),
        model="primary",
        provider="custom",
        base_url="https://primary.invalid/v1",
        _provider_fallback_tool_policy="inherit",
    )
    attempts = 0

    def fail_candidates(target, _entry, _provider, _model):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            target.model = "restricted"
            target.base_url = "https://restricted.invalid/v1"
            raise RuntimeError("first candidate failed after mutation")
        raise RuntimeError("second candidate failed before mutation")

    with (
        patch("agent.fallback_cooldown.switch_deferred_by_reset", return_value=False),
        patch("agent.fallback_cooldown._arm_rate_limit_cooldown", return_value=None),
        patch("agent.chat_completion_helpers._should_skip_fallback_candidate", return_value=False),
        patch("agent.chat_completion_helpers._fallback_chain_exhausted", return_value=False),
        patch("agent.route_binding._bind_route_entry_in_place", side_effect=fail_candidates),
    ):
        assert try_activate_fallback(agent) is False

    assert agent.model == "primary"
    assert agent.base_url == "https://primary.invalid/v1"
    assert agent._provider_fallback_tool_policy == "inherit"


@pytest.mark.parametrize(("active_policy", "target_policy"), [
    ("none", "inherit"),
    ("inherit", "none"),
])
def test_failed_primary_restore_retains_restrictive_active_policy(active_policy, target_policy):
    """A partial primary restore must not widen either route's authority."""
    from agent.agent_runtime_helpers import restore_primary_runtime

    agent = SimpleNamespace(
        _fallback_activated=True,
        _fallback_index=1,
        _primary_runtime={
            "provider": "custom", "model": "primary", "base_url": "https://primary.invalid/v1",
            "fallback_tool_policy": target_policy,
        },
        _provider_fallback_active=True,
        _provider_fallback_route=("restricted", "custom"),
        _provider_fallback_tool_policy=active_policy,
        _rate_limited_until=0,
        _restore_wait_logged=False,
        model="restricted",
        provider="custom",
        base_url="https://restricted.invalid/v1",
    )

    def partial_restore(target, *_args, **_kwargs):
        target._provider_fallback_tool_policy = target_policy
        raise RuntimeError("injected primary restore failure")

    with (
        patch("agent.fallback_cooldown._is_entitlement_rejected", return_value=False),
        patch("hermes_cli.chat_catalog.is_known_non_chat_model", return_value=False),
        patch("agent.agent_runtime_helpers._primary_quota_reopen_probe_due", return_value=False),
        patch("agent.agent_runtime_helpers._primary_reset_gate_blocks", return_value=(False, None, False)),
        patch("agent.route_binding.reinstall_primary_runtime", side_effect=partial_restore),
    ):
        assert restore_primary_runtime(agent) is False

    assert agent._provider_fallback_tool_policy == "none"


def test_failed_primary_route_preparation_does_not_advance_live_credential_pool(monkeypatch):
    """A rejected route must fail before post-commit credential selection can touch the live pool."""
    from agent import route_binding
    from agent.credential_pool import CredentialPool, PooledCredential

    pool = CredentialPool("custom", [PooledCredential(
        provider="custom", id="primary-1", label="primary", auth_type="api_key",
        priority=0, source="test", access_token="test-key", request_count=2,
    )])
    agent = SimpleNamespace(
        _credential_pool=pool,
        _credential_pool_entry_id="original",
        _provider_fallback_tool_policy="none",
    )

    monkeypatch.setattr(
        route_binding,
        "_reinstall_runtime_snapshot_in_place",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected primary route failure")),
    )

    with pytest.raises(RuntimeError, match="primary route failure"):
        route_binding.reinstall_primary_runtime(
            agent,
            {"fallback_tool_policy": "inherit"},
            "custom",
            "primary-model",
            lambda candidate: getattr(candidate, "provider", "") == "custom",
            lambda: pool,
        )

    assert pool._current_id is None
    assert [(entry.id, entry.request_count) for entry in pool.entries()] == [("primary-1", 2)]
    assert agent._credential_pool_entry_id == "original"
    assert agent._provider_fallback_tool_policy == "none"


def test_successful_primary_restore_selects_pool_only_after_route_commit(monkeypatch):
    """A successful route may select the live pool after its primary state is committed."""
    from agent import route_binding
    from agent.credential_pool import CredentialPool, PooledCredential

    pool = CredentialPool("custom", [PooledCredential(
        provider="custom", id="primary-1", label="primary", auth_type="api_key",
        priority=0, source="test", access_token="test-key", request_count=2,
    )])

    class Agent(SimpleNamespace):
        def _swap_credential(self, selected):
            self.selected = selected.id

    agent = Agent(
        _credential_pool=pool,
        _credential_pool_entry_id=None,
        _provider_fallback_active=True,
        _provider_fallback_route=("restricted", "custom"),
        _provider_fallback_tool_policy="none",
    )
    monkeypatch.setattr(
        route_binding, "_reinstall_runtime_snapshot_in_place", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "agent.agent_runtime_helpers._rebuild_primary_client", lambda *_args, **_kwargs: None,
    )

    route_binding.reinstall_primary_runtime(
        agent,
        {"fallback_tool_policy": "inherit"},
        "custom",
        "primary-model",
        lambda candidate: getattr(candidate, "provider", "") == "custom",
        lambda: pool,
    )

    assert agent._credential_pool is pool
    assert [(entry.id, entry.request_count) for entry in pool.entries()] == [("primary-1", 3)]
    assert pool.current().id == "primary-1"
    assert agent.selected == "primary-1"
    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None


def test_post_commit_pool_failure_keeps_primary_route(monkeypatch):
    """Pool maintenance failure cannot roll back or re-quarantine a committed primary route."""
    from agent import route_binding

    agent = SimpleNamespace(
        model="fallback-model",
        provider="fallback-provider",
        _provider_fallback_tool_policy="none",
        _provider_fallback_active=True,
        _provider_fallback_route=("fallback-model", "fallback-provider"),
        _fallback_activated=True,
        _fallback_index=1,
        _rate_limit_backoff_count=2,
        context_compressor=None,
    )

    def restore_route(staged, _rt, **_kwargs):
        staged.model = "primary-model"
        staged.provider = "primary-provider"
        staged._provider_fallback_tool_policy = "inherit"

    monkeypatch.setattr(route_binding, "_reinstall_runtime_snapshot_in_place", restore_route)
    monkeypatch.setattr(
        "agent.agent_runtime_helpers._rebuild_primary_client", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        route_binding,
        "rebind_primary_credential_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pool maintenance failed")),
    )

    route_binding.reinstall_primary_runtime(
        agent,
        {"fallback_tool_policy": "inherit"},
        "primary-provider",
        "primary-model",
        lambda _candidate: True,
        lambda: None,
    )

    assert agent.model == "primary-model"
    assert agent.provider == "primary-provider"
    assert agent._provider_fallback_tool_policy == "inherit"
    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None


def test_post_commit_fallback_pool_failure_keeps_committed_route(monkeypatch):
    """Fallback pool maintenance is not allowed to escape after route publication."""
    from agent import route_binding

    agent = SimpleNamespace(
        model="primary",
        provider="custom",
        base_url="https://primary.invalid/v1",
        _provider_fallback_tool_policy="inherit",
        context_compressor=None,
    )

    def prepare(staged, _entry, provider, model, *, owner=None):
        staged.model = model
        staged.provider = provider
        return "primary", "custom"

    monkeypatch.setattr(route_binding, "_bind_route_entry_in_place", prepare)
    monkeypatch.setattr("agent.chat_completion_helpers.rewrite_prompt_model_identity", lambda *_args: None)
    monkeypatch.setattr("agent.chat_completion_helpers._reset_stale_streak", lambda *_args: None)
    monkeypatch.setattr("agent.native_compaction.resolve_native_compaction_capabilities", lambda **_kwargs: {})
    monkeypatch.setattr(
        route_binding,
        "_materialize_route_client",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("pool maintenance failed")),
    )

    assert route_binding.bind_route_entry(
        agent,
        {"provider": "custom", "model": "restricted", "tool_policy": "none"},
        "custom",
        "restricted",
    ) == ("primary", "custom")
    assert agent.model == "restricted"
    assert agent._provider_fallback_active is True
    assert agent._provider_fallback_tool_policy == "none"


def test_unmaterialized_committed_route_cannot_call_provider():
    """A post-commit client failure leaves a route that is safe rather than network-capable."""
    from agent.turn_api_call import perform_api_call

    agent = SimpleNamespace(_route_client_ready=False)
    with pytest.raises(RuntimeError, match="materialisation is incomplete"):
        perform_api_call(
            agent,
            api_kwargs={},
            _original_api_kwargs={},
            _llm_middleware_trace=[],
            _moa_prepared_request=None,
            _retry=None,
            thinking_spinner=None,
            retry_count=0,
            api_call_count=0,
            api_request_id="test-request",
            effective_task_id=None,
            turn_id="test-turn",
            interrupted=False,
        )


def test_unmaterialized_committed_route_cannot_prepare_moa_or_middleware():
    """Request assembly stops before MoA preparation and other model-facing hooks."""
    from agent.turn_request_assembly import assemble_api_request

    agent = SimpleNamespace(_route_client_ready=False)
    with pytest.raises(RuntimeError, match="materialisation is incomplete"):
        assemble_api_request(
            agent,
            messages=[],
            current_turn_user_idx=-1,
            _ext_prefetch_cache=None,
            _plugin_user_context=None,
            moa_config=None,
            active_system_prompt=None,
            original_user_message=None,
            pending_moa_prepared_request=None,
            request_logger=MagicMock(),
        )


def test_context_metadata_probe_runs_only_after_fallback_publication(monkeypatch):
    """Context-window discovery may touch a provider, so candidate preparation uses static metadata."""
    from agent import route_binding

    agent = _agent()
    agent._compression_feasibility_checked = True
    events = []
    real_commit = route_binding._commit_route_agent

    def commit(target, staged):
        real_commit(target, staged)
        events.append("commit")

    def provider_probe(*_args, **_kwargs):
        assert events == ["commit"]
        events.append("probe")
        return 131_072

    def revalidate(_agent):
        assert events == ["commit", "probe"]
        events.append("feasibility")

    client = SimpleNamespace(base_url="https://fallback.invalid/v1", api_key="runtime-key")
    monkeypatch.setattr(route_binding, "_commit_route_agent", commit)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", provider_probe)
    monkeypatch.setattr(
        "agent.conversation_compression.revalidate_compression_feasibility", revalidate,
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda *_args, **_kwargs: (client, "restricted"),
    )

    assert route_binding.bind_route_entry(
        agent,
        {
            "provider": "custom",
            "model": "restricted",
            "base_url": "https://fallback.invalid/v1",
            "tool_policy": "none",
        },
        "custom",
        "restricted",
    ) == ("", "")
    assert events == ["commit", "probe", "feasibility"]


def test_nous_candidate_validation_does_not_open_auth_store():
    """Credential presence is post-commit maintenance, not candidate-admission evidence."""
    from agent.chat_completion_helpers import _fallback_entry_unavailable_without_network

    with patch("hermes_cli.auth.get_provider_auth_state", return_value={"access_token": "[REDACTED]"}) as load_auth:
        assert _fallback_entry_unavailable_without_network(
            SimpleNamespace(), {"provider": "nous", "model": "example/model"},
        ) is None
    load_auth.assert_not_called()


def test_post_commit_compressor_persistence_failure_keeps_committed_route(monkeypatch):
    """A durable-state reporting error after publication cannot make the route look rejected."""
    from agent import route_binding

    live_compressor = SimpleNamespace(_session_db=object(), _session_id="session")
    staged_compressor = SimpleNamespace(_session_db=None, _session_id="")
    agent = SimpleNamespace(model="primary", context_compressor=live_compressor)
    staged = SimpleNamespace(model="restricted", context_compressor=staged_compressor)
    monkeypatch.setattr(
        "agent.route_binding_state.persist_prepared_compressor",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("persistence failed")),
    )

    route_binding._commit_route_agent(agent, staged)

    assert agent.model == "restricted"
    assert agent.context_compressor is staged_compressor


@pytest.mark.parametrize(("active_policy", "target_policy"), [
    ("none", "inherit"),
    ("inherit", "none"),
])
def test_runtime_snapshot_restore_is_transactional_on_late_failure(
    monkeypatch, active_policy, target_policy,
):
    """Identity/client stay paired; only the fail-closed policy clamp may publish on failure."""
    from agent import route_binding

    old_client = object()
    agent = SimpleNamespace(
        model="active",
        provider="custom",
        base_url="https://active.invalid/v1",
        client=old_client,
        _provider_fallback_tool_policy=active_policy,
    )
    target = {
        "model": "target",
        "provider": "custom",
        "base_url": "https://target.invalid/v1",
        "fallback_tool_policy": target_policy,
    }

    def fail_after_partial_stage(staged, _snapshot, **_kwargs):
        staged.model = "target"
        staged.base_url = "https://target.invalid/v1"
        staged.client = object()
        staged._provider_fallback_tool_policy = target_policy
        raise RuntimeError("injected late restore failure")

    monkeypatch.setattr(route_binding, "_reinstall_runtime_snapshot_in_place", fail_after_partial_stage)

    with pytest.raises(RuntimeError, match="late restore failure"):
        route_binding.reinstall_runtime_snapshot(agent, target)

    assert agent.model == "active"
    assert agent.base_url == "https://active.invalid/v1"
    assert agent.client is old_client
    assert agent._provider_fallback_tool_policy == "none"


def test_middleware_cannot_reintroduce_tools_on_response_only_fallback():
    """The final provider-call boundary strips schemas added by request middleware."""
    from agent.turn_api_call import perform_api_call

    sent = {}

    def api_call(kwargs):
        sent.update(kwargs)
        return SimpleNamespace()

    agent = SimpleNamespace(
        api_mode="chat_completions",
        _disable_streaming=True,
        base_url="https://restricted.invalid/v1",
        provider="custom",
        model="restricted",
        client=object(),
        session_id="",
        platform="",
        _provider_fallback_tool_policy="none",
        _model_request_active=None,
        _pending_redirect_lock=None,
        _pending_redirect=None,
        _has_stream_consumers=lambda: False,
        _has_pending_redirect=lambda: False,
        _interruptible_api_call=api_call,
    )

    def inject_tools(request, execute, **_kwargs):
        poisoned = dict(request)
        poisoned["tools"] = [{"type": "function", "function": {"name": "terminal"}}]
        poisoned["tool_choice"] = "required"
        return execute(poisoned)

    with (
        patch("hermes_cli.middleware.run_llm_execution_middleware", side_effect=inject_tools),
        patch("agent.relay_llm.execute", side_effect=lambda request, callback, **_kwargs: callback(request)),
    ):
        perform_api_call(
            agent,
            api_kwargs={"model": "restricted", "messages": []},
            _original_api_kwargs={},
            _llm_middleware_trace=[],
            _moa_prepared_request=None,
            _retry=SimpleNamespace(),
            thinking_spinner=None,
            retry_count=0,
            api_call_count=0,
            api_request_id="r1",
            effective_task_id="",
            turn_id="t1",
            interrupted=False,
        )

    assert "tools" not in sent
    assert "tool_choice" not in sent
