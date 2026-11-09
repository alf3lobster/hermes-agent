from pathlib import Path


def _provider(config=None):
    from plugins.memory.holographic import HolographicMemoryProvider

    return HolographicMemoryProvider(config or {})


def test_format_conversation_excludes_compaction_summary_messages(monkeypatch):
    from agent import context_compressor

    provider = _provider()
    summary = {"role": "user", "content": "generated compaction summary"}
    ordinary = {"role": "user", "content": "I prefer concise reports"}
    monkeypatch.setattr(
        context_compressor,
        "is_compaction_summary_message",
        lambda message: message is summary,
    )

    rendered = provider._format_conversation([summary, ordinary], max_turns=80)

    assert "generated compaction summary" not in rendered
    assert "I prefer concise reports" in rendered


def test_save_config_uses_raw_complete_state_and_atomic_writer(tmp_path, monkeypatch):
    from hermes_cli import config as config_mod

    home = tmp_path / "profile"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text("model:\n  default: kept\nplugins:\n  sibling:\n    enabled: true\n", encoding="utf-8")
    calls = []
    real_write = config_mod.atomic_config_write

    def recording_write(config_path, data):
        calls.append((config_path, data))
        real_write(config_path, data)

    monkeypatch.setattr(config_mod, "atomic_config_write", recording_write)
    provider = _provider()
    provider.save_config({"auto_extract": "false"}, home)

    assert calls and calls[0][0] == path
    written = config_mod.read_user_config_raw(path)
    assert written["model"]["default"] == "kept"
    assert written["plugins"]["sibling"]["enabled"] is True
    assert written["plugins"]["hermes-memory-store"]["auto_extract"] == "false"


def test_string_false_disables_per_turn_extraction():
    provider = _provider({"auto_extract": "false", "auto_extract_trigger": "per_turn"})
    provider.sync_turn("remember this", "ack", session_id="s")
    assert provider._turn_counts == {}


def test_session_dry_run_blocks_short_conversation_regex_writes(tmp_path):
    provider = _provider({
        "db_path": str(tmp_path / "memory.db"),
        "auto_extract_dry_run": True,
        "auto_extract_min_turns": 5,
    })
    provider.initialize("session")

    provider._auto_extract_facts([
        {"role": "user", "content": "I prefer concise updates with clear evidence."},
    ])

    assert provider._store is not None
    count = provider._store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert count == 0


def test_session_dry_run_blocks_regex_writes_after_model_error(tmp_path, monkeypatch):
    provider = _provider({
        "db_path": str(tmp_path / "memory.db"),
        "auto_extract_dry_run": True,
        "auto_extract_min_turns": 2,
    })
    provider.initialize("session")
    monkeypatch.setattr(
        provider,
        "_call_extraction_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )

    provider._auto_extract_facts([
        {"role": "user", "content": "I prefer concise updates with clear evidence."},
        {"role": "assistant", "content": "Understood."},
    ])

    assert provider._store is not None
    count = provider._store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert count == 0
