"""Tests for post-publication Nous fallback credential maintenance."""

from __future__ import annotations

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = None
        return agent


def _mock_client(base_url="https://chatgpt.com/backend-api/codex", api_key="fb-key"):
    mock = type("Client", (), {})()
    mock.base_url = base_url
    mock.api_key = api_key
    mock.chat = type("Chat", (), {})()
    mock.chat.completions = type("Completions", (), {})()
    mock.chat.completions.create = lambda *args, **kwargs: None
    return mock


class TestNousFallbackLocalAvailability:
    def test_missing_nous_token_does_not_skip_candidate_before_publication(self):
        """Credential presence cannot choose a different candidate before commit."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        with patch(
            "hermes_cli.auth.get_provider_auth_state",
            return_value={},
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(api_key="fb"), "gpt-5.5"),
        ):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.provider == "nous"
        assert agent.model == "anthropic/claude-sonnet-4.6"


    def test_present_nous_token_allows_activation(self):
        """Nous is considered when token material exists."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        with patch(
            "hermes_cli.auth.get_provider_auth_state",
            return_value={"access_token": "abc", "refresh_token": "xyz"},
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(api_key="fb"), "anthropic/claude-sonnet-4.6"),
        ):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.provider == "nous"
