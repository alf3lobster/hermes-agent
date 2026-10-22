"""Transactional readiness regressions for in-place operator model switches."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import (
    _restore_switch_snapshot,
    _snapshot_switch_state,
    switch_model,
)
from tests.agent.test_switch_model_pool_reload import _make_agent, _make_pool


def _switch(agent):
    with patch(
        "agent.agent_runtime_helpers._resolve_switch_context_length",
        return_value=([], None),
    ), patch(
        "agent.agent_runtime_helpers._apply_switched_provider_request_overrides",
        return_value=None,
    ):
        switch_model(
            agent,
            new_model="llama-3.3-70b",
            new_provider="groq",
            api_key="groq-key-new",
            base_url="https://api.groq.com/openai/v1",
            api_mode="chat_completions",
        )


def _assert_prior_route(agent, old_pool):
    assert agent.model == "qwen-coder"
    assert agent.provider == "opencode-go"
    assert agent._credential_pool is old_pool
    assert agent._route_client_ready is True
    assert agent._provider_fallback_tool_policy == "none"
    assert agent._primary_runtime == {"route": "old"}
    assert agent._primary_credential_pool is old_pool


def _transaction_agent():
    old_pool = _make_pool("opencode-go")
    agent = _make_agent("opencode-go", "qwen-coder", old_pool)
    agent._route_client_ready = True
    agent._provider_fallback_tool_policy = "none"
    agent._primary_runtime = {"route": "old"}
    agent._primary_credential_pool = old_pool
    return agent, old_pool


def test_snapshot_rollback_preserves_absent_route_fields():
    agent = SimpleNamespace(model="old")
    snapshot = _snapshot_switch_state(agent)
    agent._route_client_ready = False
    agent._moa_slot_runtimes = {"new": object()}
    agent._transport_cache = {"new": object()}

    _restore_switch_snapshot(agent, snapshot)

    assert not hasattr(agent, "_route_client_ready")
    assert not hasattr(agent, "_moa_slot_runtimes")
    assert not hasattr(agent, "_transport_cache")


def test_primary_snapshot_failure_restores_exact_prior_route():
    agent, old_pool = _transaction_agent()
    new_pool = _make_pool("groq")

    with patch("agent.credential_pool.load_pool", return_value=new_pool), patch(
        "agent.agent_runtime_helpers._build_primary_runtime_snapshot",
        side_effect=RuntimeError("snapshot failed"),
    ), pytest.raises(RuntimeError, match="snapshot failed"):
        _switch(agent)

    _assert_prior_route(agent, old_pool)


def test_billing_persistence_failure_restores_exact_prior_route():
    agent, old_pool = _transaction_agent()
    new_pool = _make_pool("groq")
    agent._session_db = type(
        "FailingSessionDB",
        (),
        {"update_session_billing_route": lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("billing failed"))},
    )()
    agent.session_id = "session-1"

    with patch("agent.credential_pool.load_pool", return_value=new_pool), pytest.raises(
        RuntimeError, match="billing failed"
    ):
        _switch(agent)

    _assert_prior_route(agent, old_pool)


def test_compressor_persistence_failure_restores_original_compressor():
    agent, old_pool = _transaction_agent()
    old_db = object()
    original_compressor = SimpleNamespace(
        model="qwen-coder", _session_db=old_db, _session_id="session-1"
    )
    agent.context_compressor = original_compressor
    new_pool = _make_pool("groq")
    readiness_seen = []

    def update_compressor(target, *_args):
        target.context_compressor.model = "llama-3.3-70b"

    def fail_persistence(previous, prepared):
        readiness_seen.append(target_ready := agent._route_client_ready)
        assert target_ready is False
        assert previous is original_compressor
        assert prepared is not original_compressor
        assert prepared._session_db is old_db
        raise RuntimeError("compressor persistence failed")

    with patch("agent.credential_pool.load_pool", return_value=new_pool), patch(
        "agent.agent_runtime_helpers._update_switch_compressor", update_compressor
    ), patch(
        "agent.agent_runtime_helpers._build_primary_runtime_snapshot",
        return_value={"request_overrides": {}},
    ), patch(
        "agent.route_binding_state.persist_prepared_compressor", fail_persistence
    ), pytest.raises(RuntimeError, match="compressor persistence failed"):
        _switch(agent)

    _assert_prior_route(agent, old_pool)
    assert agent.context_compressor is original_compressor
    assert original_compressor.model == "qwen-coder"
    assert readiness_seen == [False]


def test_readiness_is_false_during_snapshot_and_persistence():
    agent, _old_pool = _transaction_agent()
    new_pool = _make_pool("groq")
    readiness_seen = []

    def snapshot(target, api_mode):
        readiness_seen.append(("snapshot", target._route_client_ready))
        return {"request_overrides": {}, "api_mode": api_mode}

    def persist(target):
        readiness_seen.append(("persist", target._route_client_ready))

    with patch("agent.credential_pool.load_pool", return_value=new_pool), patch(
        "agent.agent_runtime_helpers._build_primary_runtime_snapshot", snapshot
    ), patch("agent.agent_runtime_helpers._persist_switch_billing_route", persist):
        _switch(agent)

    assert readiness_seen == [("snapshot", False), ("persist", False)]
    assert agent._route_client_ready is True
    assert agent._provider_fallback_tool_policy == "inherit"
