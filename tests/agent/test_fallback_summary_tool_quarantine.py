"""Response-only fallback regressions for independently dispatched summary requests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import (
    _anthropic_summary_attempt,
    _chat_summary_attempt,
    direct_api_call,
)
from run_agent import AIAgent
from tests.agent.test_run_agent import _make_tool_defs, _mock_response


def _agent() -> AIAgent:
    with (
        patch("hermes_cli.plugins.discover_plugins", return_value=None),
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._provider_fallback_tool_policy = "none"
    return agent


def test_response_only_fallback_summary_omits_tool_schemas():
    """The iteration-summary side path cannot reacquire fallback tool schemas."""
    agent = _agent()
    agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
    assert agent.tools

    result = agent._handle_max_iterations(
        [{"role": "user", "content": "do stuff"}], 60
    )

    assert result == "Summary"
    sent = agent.client.chat.completions.create.call_args.kwargs
    assert not sent.get("tools")
    assert "tool_choice" not in sent
    assert "parallel_tool_calls" not in sent


def test_inert_route_refuses_iteration_summary_provider_call():
    """No independent summary dispatch can use a client whose route is incomplete."""
    agent = _agent()
    agent._route_client_ready = False
    agent.client.chat.completions.create.return_value = _mock_response(content="UNEXPECTED")

    result = agent._handle_max_iterations(
        [{"role": "user", "content": "do stuff"}], 60
    )

    assert result != "UNEXPECTED"
    agent.client.chat.completions.create.assert_not_called()


def test_inert_route_refuses_direct_provider_call():
    """Cron/delegated inline dispatch shares the same readiness fence."""
    agent = _agent()
    agent._route_client_ready = False

    with pytest.raises(RuntimeError, match="route client is not ready"):
        direct_api_call(agent, {"model": "test", "messages": []})

    agent.client.chat.completions.create.assert_not_called()


def test_inert_moa_route_refuses_summary_prepare():
    """MoA advisor fan-out is fenced before prepare(), not merely before final dispatch."""
    agent = _agent()
    agent.provider = "moa"
    agent._route_client_ready = False
    agent.client.chat.completions.prepare = MagicMock()

    with pytest.raises(RuntimeError, match="route client is not ready"):
        _chat_summary_attempt(
            agent,
            [{"role": "user", "content": "summarise"}],
            "iteration-summary:test",
        )

    agent.client.chat.completions.prepare.assert_not_called()


def test_anthropic_summary_refuses_before_relay_and_keeps_ready_path():
    """Native Anthropic summaries fence relay registration, not only its callback."""
    agent = _agent()
    agent.api_mode = "anthropic_messages"
    agent._is_anthropic_oauth = False
    agent._anthropic_base_url = "https://api.anthropic.com"
    transport = MagicMock()
    transport.build_kwargs.return_value = {"model": agent.model, "messages": []}
    transport.normalize_response.return_value = SimpleNamespace(
        content="Summary", tool_calls=[]
    )
    agent._get_transport = MagicMock(return_value=transport)

    with patch(
        "agent.relay_llm.execute_current", return_value=SimpleNamespace()
    ) as execute_current:
        agent._route_client_ready = False
        with pytest.raises(RuntimeError, match="route client is not ready"):
            _anthropic_summary_attempt(agent, [], "iteration-summary:test")(0)
        execute_current.assert_not_called()

        agent._route_client_ready = True
        assert (
            _anthropic_summary_attempt(agent, [], "iteration-summary:test")(0)
            == "Summary"
        )
        execute_current.assert_called_once()
