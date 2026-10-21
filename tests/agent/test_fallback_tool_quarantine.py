"""Fail-closed dispatch tests for response-only fallback routes."""

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
        patch("agent.agent_runtime_helpers._primary_quota_reopened_early", return_value=False),
        patch("agent.agent_runtime_helpers._primary_reset_gate_blocks", return_value=(False, None, False)),
        patch("agent.route_binding.reinstall_primary_runtime", side_effect=partial_restore),
    ):
        assert restore_primary_runtime(agent) is False

    assert agent._provider_fallback_tool_policy == "none"


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

    def fail_after_partial_stage(staged, _snapshot):
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
