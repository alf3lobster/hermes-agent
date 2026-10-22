"""Response-only fallback regressions for independently dispatched summary requests."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tests.agent.test_run_agent import _make_tool_defs, _mock_response


def test_response_only_fallback_summary_omits_tool_schemas():
    """The iteration-summary side path cannot reacquire fallback tool schemas."""
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
    agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
    agent._cached_system_prompt = "You are helpful."
    agent._provider_fallback_tool_policy = "none"
    assert agent.tools

    result = agent._handle_max_iterations(
        [{"role": "user", "content": "do stuff"}], 60
    )

    assert result == "Summary"
    sent = agent.client.chat.completions.create.call_args.kwargs
    assert not sent.get("tools")
    assert "tool_choice" not in sent
    assert "parallel_tool_calls" not in sent
