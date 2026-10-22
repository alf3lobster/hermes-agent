"""``auxiliary.voice_chat``: a voice turn runs on the voice model, the next turn on the main one.

Two real loopback providers; the agent talks HTTP to both, so the assertion is on what each
server actually received rather than on agent attributes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from run_agent import AIAgent
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home


def _agent(home: Path, main_url: str):
    return AIAgent(
        model="fake-model", provider="custom", base_url=main_url, api_key="sk-fake-e2e",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
    )


@pytest.mark.parametrize("voice_window_ok", [True, False])
def test_voice_turn_routes_then_restores(tmp_path, monkeypatch, voice_window_ok):
    with FakeLLMServer([Text("main one"), Text("main two")]) as main, \
            FakeLLMServer([Text("voice reply")]) as voice:
        context = "128000" if voice_window_ok else "1000"
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: sk-fake-voice\n"
            "custom_providers:\n  - name: voice\n"
            f"    base_url: {voice.base_url}\n    models:\n      voice-model:\n        context_length: {context}\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)

        first = agent.run_conversation("typed question")
        agent._voice_turn_pending = True
        spoken = agent.run_conversation("spoken question", conversation_history=first["messages"])
        agent.run_conversation("typed again", conversation_history=spoken["messages"])

        main_models = [r["model"] for r in main.main_requests()]
        voice_models = [r["model"] for r in voice.main_requests()]
        if voice_window_ok:
            assert voice_models == ["voice-model"]
            # Unconfigured effort: the voice turn goes out with reasoning off, the main turns untouched.
            assert voice.main_requests()[0]["reasoning_effort"] == "none"
            assert {r["reasoning_effort"] for r in main.main_requests()} == {"medium"}
            assert main_models == ["fake-model", "fake-model"]
            assert spoken["model"] == "voice-model"
        else:  # too large for the voice model's window: the main model answers, nothing compacts
            assert voice_models == []
            assert main_models == ["fake-model"] * 3
        assert agent.model == "fake-model"
        assert agent.base_url.rstrip("/") == main.base_url.rstrip("/")
        assert agent._fallback_activated is False


def test_undersized_voice_route_is_rejected_before_publication(tmp_path, monkeypatch):
    """A route that cannot fit the turn must never become live and therefore needs no restore."""
    from agent.voice_turn_route import begin_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
            "custom_providers:\n  - name: voice\n"
            f"    base_url: {voice.base_url}\n    models:\n      voice-model:\n        context_length: 1000\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        original_client = agent.client
        original_compressor = agent.context_compressor
        agent._voice_turn_pending = True
        monkeypatch.setattr(
            "agent.route_binding.reinstall_runtime_snapshot",
            lambda *_args: (_ for _ in ()).throw(AssertionError("rejected route required restore")),
        )

        begin_voice_turn_route(
            agent,
            [{"role": "user", "content": "large " * 5000}],
            "system",
        )

        assert agent.model == "fake-model"
        assert agent.client is original_client
        assert agent.context_compressor is original_compressor
        assert agent._fallback_activated is False


def test_inert_committed_voice_route_is_not_reported_as_routed(tmp_path, monkeypatch, caplog):
    """A committed voice identity with failed maintenance remains inert and is not success-labelled."""
    from agent.voice_turn_route import begin_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent._voice_turn_pending = True

        def commit_inert(target, *_args, **_kwargs):
            target.model = "voice-model"
            target.provider = "custom"
            target._route_client_ready = False
            return "fake-model", "custom"

        monkeypatch.setattr("agent.route_binding.bind_route_entry", commit_inert)
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

        assert agent.model == "voice-model"
        assert agent._route_client_ready is False
        assert getattr(agent, "_turn_route_task", None) is None
        assert "Voice turn routed" not in caplog.text


def test_post_commit_voice_maintenance_failure_is_not_reported_as_main_fallback(
    tmp_path, monkeypatch, caplog,
):
    """Voice publication failure retains the inert voice route without a false main-route notice."""
    from agent.route_binding import RouteCommittedError
    from agent.voice_turn_route import begin_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent._voice_turn_pending = True

        def commit_then_fail(target, *_args, **_kwargs):
            target.model = "voice-model"
            target.provider = "custom"
            target._fallback_activated = True
            target._route_client_ready = False
            raise RouteCommittedError("committed route maintenance failed")

        monkeypatch.setattr("agent.route_binding.bind_route_entry", commit_then_fail)
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

        assert agent.model == "voice-model"
        assert agent._route_client_ready is False
        assert agent._fallback_activated is False
        assert agent._voice_route_state == {"reasoning_config": None}
        assert "not used" not in caplog.text
        assert "using fake-model" not in caplog.text
        assert "Voice turn routed" not in caplog.text


def test_end_voice_post_commit_restore_failure_does_not_arm_retry(tmp_path, monkeypatch):
    """A real published session route stays inert and cannot trigger next-turn restoration."""
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent._fallback_chain = [{"provider": "custom", "model": "session-fallback"}]
        agent._fallback_index = 1
        agent._fallback_activated = True
        agent._provider_fallback_active = True
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")
        assert agent.model == "voice-model"
        assert agent._fallback_activated is True
        assert agent._route_client_ready is True

        monkeypatch.setattr(
            "agent.route_binding_state.persist_prepared_compressor",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("db unavailable")),
        )
        end_voice_turn_route(agent)

        assert agent.model == "fake-model"
        assert agent._route_client_ready is False
        assert agent._fallback_activated is False
        assert agent._voice_route_state is None
        assert agent._restore_primary_runtime() is False


def test_end_voice_client_rebuild_failure_closes_restore_bookkeeping(tmp_path, monkeypatch):
    """A committed session route with failed client rebuild cannot re-arm the prior fallback."""
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent._fallback_chain = [{"provider": "custom", "model": "session-fallback"}]
        agent._fallback_index = 1
        agent._fallback_activated = True
        agent._provider_fallback_active = True
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

        monkeypatch.setattr(
            "agent.agent_runtime_helpers._rebuild_primary_client",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("client rebuild failed")),
        )
        end_voice_turn_route(agent)

        assert agent.model == "fake-model"
        assert agent._route_client_ready is False
        assert agent._fallback_activated is False
        assert agent._fallback_index == len(agent._fallback_chain)
        assert agent._restore_primary_runtime() is False


def test_committed_inert_route_cannot_enter_voice_route(tmp_path, monkeypatch):
    """An ordinary voice turn is not authority to leave a committed inert restrictive route."""
    from agent.voice_turn_route import begin_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent.model = "restricted"
        agent._provider_fallback_tool_policy = "none"
        agent._route_client_ready = False
        agent._fallback_chain = [{"provider": "custom", "model": "restricted"}]
        agent._fallback_index = 1
        agent._fallback_activated = False
        agent._voice_turn_pending = True
        monkeypatch.setattr(
            "agent.route_binding.bind_route_entry",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inert route escaped")),
        )

        result = begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

        assert result == "system"
        assert agent.model == "restricted"
        assert agent._provider_fallback_tool_policy == "none"
        assert agent._route_client_ready is False
        assert agent._fallback_index == 1


def test_voice_restore_rebinds_captured_pool_before_readiness(tmp_path, monkeypatch):
    """The session pool is part of the atomic runtime snapshot, never a post-ready extra."""
    import agent.voice_turn_route as voice_route
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    from types import SimpleNamespace

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: openrouter\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        session_pool = SimpleNamespace(
            provider="custom",
            entry_id_for_api_key=lambda _key: "session-entry",
        )
        voice_pool = SimpleNamespace(
            provider="openrouter",
            has_credentials=lambda: True,
            entry_id_for_api_key=lambda _key: "voice-entry",
        )
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: voice_pool)
        agent._credential_pool = session_pool
        agent._credential_pool_entry_id = "session-entry"
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")
        assert agent._credential_pool is not session_pool

        observed = []
        original_restore = voice_route._restore_extra_fields

        def observe_restore(target, state):
            observed.append((
                target._route_client_ready,
                target._credential_pool is session_pool,
                target._credential_pool_entry_id,
            ))
            original_restore(target, state)

        monkeypatch.setattr(voice_route, "_restore_extra_fields", observe_restore)
        end_voice_turn_route(agent)

        assert observed == [(True, True, "session-entry")]
        assert "_credential_pool" not in voice_route._EXTRA_FIELDS
        assert agent._credential_pool is session_pool


@pytest.mark.parametrize("model,api_mode,expected", [
    ("gpt-6-astra", "codex_responses", "low"),        # Responses ladder has no "none"
    ("claude-opus-5-5", "anthropic_messages", "low"),  # mandatory thinking
    ("gpt-5.6-sol", "codex_responses", None),          # "none" is on its ladder: stays off
    ("claude-sonnet-4-6", "anthropic_messages", None),  # accepts thinking.type=disabled
])
def test_reasoning_off_falls_to_the_lowest_valid_level(model, api_mode, expected):
    from types import SimpleNamespace

    from agent.voice_turn_route import _voice_reasoning

    agent = SimpleNamespace(model=model, api_mode=api_mode, provider="custom",
                            base_url="https://api.openai.com/v1" if "gpt" in model else "https://api.anthropic.com")
    got = _voice_reasoning(agent, {"enabled": False})
    assert got == ({"enabled": True, "effort": expected} if expected else {"enabled": False})


def test_voice_usage_never_becomes_the_session_route(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli", model="main-model")
    db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="voice-model",
                           billing_provider="voicep", api_call_count=1, task="voice_chat")
    row = db.get_session("s1")
    assert row["model"] == "main-model"
    assert row["input_tokens"] == 10
    assert db.auxiliary_usage_by_task("s1")["voice_chat"]["input_tokens"] == 10
    assert db.get_recent_session_model_route("s1") is None


@pytest.mark.parametrize("separate_voice_model", [False, True])
def test_mid_voice_turn_reports_the_sessions_own_route(tmp_path, monkeypatch, separate_voice_model):
    """Desktop adopts session.info's effort/model into the composer that seeds new chats: a live
    voice turn's reasoning-off (and voice model) must never be reported or persisted as the chat's."""
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    from tui_gateway.server import _runtime_model_config, _session_info

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        extra = ("auxiliary:\n  voice_chat:\n    provider: custom\n"
                 f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: sk-fake-voice\n"
                 if separate_voice_model else "")
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=extra)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        agent._provider_fallback_tool_policy = "none"
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")
        try:
            assert agent.reasoning_config == {"enabled": False}  # the voice turn itself still runs off
            assert agent.model == ("voice-model" if separate_voice_model else "fake-model")
            assert agent._provider_fallback_tool_policy == (
                "inherit" if separate_voice_model else "none"
            )
            info = _session_info(agent, {})
            persisted = _runtime_model_config(agent)
            assert (info["reasoning_effort"], info["model"]) == ("low", "fake-model")
            assert persisted["reasoning_config"] == {"enabled": True, "effort": "low"}
            assert persisted["model"] == "fake-model"
        finally:
            end_voice_turn_route(agent)
        assert agent._provider_fallback_tool_policy == "none"
        assert _session_info(agent, {})["reasoning_effort"] == "low"


def test_failed_voice_bind_leaves_the_complete_session_route_untouched(tmp_path, monkeypatch):
    """A late voice bind failure must not need a second, fallible restore transition."""
    from agent import chat_completion_helpers, route_binding
    from agent.voice_turn_route import begin_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
            "    tool_policy: none\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent._provider_fallback_tool_policy = "inherit"
        agent._voice_turn_pending = True

        original_client = agent.client
        original_compressor = agent.context_compressor
        original_compressor_state = vars(original_compressor).copy()

        def fail_after_identity_client_and_compressor_mutation(*_args, **_kwargs):
            raise RuntimeError("injected late bind failure")

        def forbidden_restore(*_args, **_kwargs):
            raise AssertionError("failed atomic bind must not require runtime restoration")

        monkeypatch.setattr(
            chat_completion_helpers,
            "_rescope_fallback_extra_body",
            fail_after_identity_client_and_compressor_mutation,
        )
        monkeypatch.setattr(route_binding, "reinstall_runtime_snapshot", forbidden_restore)

        prompt = begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

        assert prompt
        assert agent.model == "fake-model"
        assert agent.provider == "custom"
        assert agent.base_url.rstrip("/") == main.base_url.rstrip("/")
        assert agent.client is original_client
        assert agent.context_compressor is original_compressor
        assert vars(agent.context_compressor) == original_compressor_state
        assert agent._provider_fallback_tool_policy == "inherit"
        assert agent._fallback_activated is False
        assert agent._voice_route_state == {
            "reasoning_config": agent.reasoning_config,
        }


def test_failed_voice_bind_does_not_write_compressor_session_state(tmp_path, monkeypatch):
    """Preparing a rejected route must not persist compressor resets for a route that never ran."""
    from agent import chat_completion_helpers
    from agent.voice_turn_route import begin_voice_turn_route

    class RecordingStore:
        def __init__(self):
            self.writes = []

        def __getattr__(self, name):
            if name.startswith(("set_", "clear_", "record_", "patch_")):
                def record(*args):
                    self.writes.append((name, args))
                return record
            raise AttributeError(name)

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
            "    tool_policy: none\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        store = RecordingStore()
        agent.context_compressor._session_db = store
        agent.context_compressor._session_id = "session-1"
        agent.context_compressor._ineffective_compression_count = 3
        agent._voice_turn_pending = True

        def fail_after_compressor_update(*_args, **_kwargs):
            raise RuntimeError("injected late bind failure")

        monkeypatch.setattr(
            chat_completion_helpers,
            "_rescope_fallback_extra_body",
            fail_after_compressor_update,
        )

        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

        assert store.writes == []
        assert agent.model == "fake-model"
        assert agent.context_compressor._ineffective_compression_count == 3


@pytest.mark.parametrize(("session_policy", "voice_policy"), [
    ("none", "inherit"),
    ("inherit", "none"),
])
def test_failed_voice_restore_keeps_the_stricter_tool_policy(
    tmp_path, monkeypatch, session_policy, voice_policy,
):
    """A partial route restore must retain the stricter policy until route restoration completes."""
    from agent import agent_runtime_helpers
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: «redacted:sk-…»\n"
            f"    tool_policy: {voice_policy}\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent._provider_fallback_tool_policy = session_policy
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")
        assert agent._provider_fallback_tool_policy == voice_policy

        def fail_after_route_restore_starts(*_args, **_kwargs):
            raise RuntimeError("injected partial restore")

        monkeypatch.setattr(agent_runtime_helpers, "_apply_primary_runtime_fields", fail_after_route_restore_starts)
        end_voice_turn_route(agent)
        assert agent._provider_fallback_tool_policy == "none"
