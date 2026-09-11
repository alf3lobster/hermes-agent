"""Compression health alarms stay bounded and content-free."""

import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import _emit_compression_attempt_telemetry


def _agent(telemetry):
    if telemetry is not None:
        telemetry = {"attempt_id": "attempt-health", **telemetry}
    compressor = SimpleNamespace(
        _last_compression_telemetry=telemetry,
        _last_summary_fallback_used=False,
        _last_aux_model_failure_model=None,
    )
    return SimpleNamespace(
        context_compressor=compressor,
        session_id="session-health",
        _compression_attempt_id="attempt-health",
    )


def _emit(telemetry, *, elapsed_seconds=1.0):
    with patch(
        "agent.conversation_compression.time.monotonic",
        return_value=100.0 + elapsed_seconds,
    ):
        _emit_compression_attempt_telemetry(
            _agent(telemetry),
            started_at=100.0,
            commit_status="committed",
            split_status="committed",
        )


def test_warns_when_compression_uses_more_than_one_auxiliary_call(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        _emit({"aux_call_count": 2})

    assert "auxiliary_calls=2>1" in caplog.text


def test_warns_when_compression_exceeds_sixty_seconds(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        _emit({"aux_call_count": 1}, elapsed_seconds=61.0)

    assert "duration_ms=61000>60000" in caplog.text


def test_warns_when_committed_compression_reclaims_no_tokens(caplog):
    telemetry = {
        "aux_call_count": 1,
        "pre_message_tokens": 10_000,
        "post_message_tokens": 10_000,
    }
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        _emit(telemetry)

    assert "token_reduction=0<=0" in caplog.text


def test_committed_token_alarm_uses_final_commit_boundary_counts(caplog):
    telemetry = {
        "aux_call_count": 1,
        "pre_message_tokens": 10_000,
        "post_message_tokens": 2_000,
    }
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        with patch(
            "agent.conversation_compression.time.monotonic",
            return_value=101.0,
        ):
            _emit_compression_attempt_telemetry(
                _agent(telemetry),
                started_at=100.0,
                commit_status="committed",
                split_status="committed",
                committed_pre_message_tokens=10_000,
                committed_post_message_tokens=10_000,
            )

    assert "token_reduction=0<=0" in caplog.text


def test_stale_attempt_telemetry_is_not_reused(caplog):
    stale = {
        "attempt_id": "old-attempt",
        "aux_call_count": 9,
        "pre_message_tokens": 10_000,
        "post_message_tokens": 20_000,
    }
    agent = _agent(None)
    agent.context_compressor._last_compression_telemetry = stale
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        with patch(
            "agent.conversation_compression.time.monotonic",
            return_value=101.0,
        ):
            _emit_compression_attempt_telemetry(
                agent,
                started_at=100.0,
                commit_status="aborted",
                split_status="aborted",
                failure_class="pool_saturated",
            )

    assert "auxiliary_calls=9>1" not in caplog.text


def test_health_alarm_reaches_user_warning_channel():
    warnings = []
    agent = _agent({"aux_call_count": 2})
    agent._emit_warning = warnings.append

    with patch(
        "agent.conversation_compression.time.monotonic",
        return_value=101.0,
    ):
        _emit_compression_attempt_telemetry(
            agent,
            started_at=100.0,
            commit_status="committed",
            split_status="committed",
        )

    assert warnings == [
        "⚠ Compression health alarm: auxiliary_calls=2>1"
    ]


def test_healthy_compression_emits_no_health_warning(caplog):
    telemetry = {
        "aux_call_count": 1,
        "pre_message_tokens": 10_000,
        "post_message_tokens": 2_000,
    }
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        _emit(telemetry, elapsed_seconds=20.0)

    assert "compression health alarm" not in caplog.text


def test_auxiliary_call_telemetry_counts_every_request():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    telemetry = compressor._begin_compression_telemetry(current_tokens=50_000)

    compressor._record_aux_compression_call(
        prompt_messages=[{"role": "user", "content": "one"}],
        max_tokens=100,
        duration_ms=10,
    )
    compressor._record_aux_compression_call(
        prompt_messages=[{"role": "user", "content": "two"}],
        max_tokens=100,
        duration_ms=20,
    )

    assert telemetry["aux_call_count"] == 2
    assert telemetry["aux_call_duration_ms"] == 30


def test_summary_pre_dispatch_failure_counts_zero_physical_requests():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    telemetry = compressor._begin_compression_telemetry(
        current_tokens=50_000,
        attempt_id="pre-dispatch",
    )

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=RuntimeError("provider resolution failed"),
    ):
        assert compressor._generate_summary(
            [{"role": "user", "content": "hello"}]
        ) is None

    assert telemetry["aux_call_count"] == 0


def test_late_auxiliary_completion_cannot_charge_new_attempt():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    old = compressor._begin_compression_telemetry(
        current_tokens=50_000,
        attempt_id="old",
    )
    old_attempt_id = old["attempt_id"]
    compressor._begin_compression_telemetry(
        current_tokens=50_000,
        attempt_id="new",
    )

    compressor._record_aux_compression_call(
        prompt_messages=[{"role": "user", "content": "late"}],
        max_tokens=100,
        duration_ms=10,
        attempt_id=old_attempt_id,
    )

    assert compressor._last_compression_telemetry["attempt_id"] == "new"
    assert compressor._last_compression_telemetry["aux_call_count"] == 0


def test_parallel_attempts_keep_auxiliary_counts_separate():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    barrier = threading.Barrier(2)
    results = {}

    def run(attempt_id):
        telemetry = compressor._begin_compression_telemetry(
            current_tokens=50_000,
            attempt_id=attempt_id,
        )
        barrier.wait()
        compressor._record_aux_compression_call(
            prompt_messages=[{"role": "user", "content": attempt_id}],
            max_tokens=100,
            duration_ms=10,
            attempt_id=attempt_id,
        )
        results[attempt_id] = dict(telemetry)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results["a"]["aux_call_count"] == 1
    assert results["b"]["aux_call_count"] == 1


def test_parallel_attempt_seeds_are_execution_context_local():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    barrier = threading.Barrier(2)
    results = {}

    def run(attempt_id):
        compressor._seed_compression_telemetry(
            attempt_id=attempt_id,
            session_id="shared-session",
            trigger_source="auto",
        )
        barrier.wait()
        telemetry = compressor._begin_compression_telemetry(current_tokens=50_000)
        results[attempt_id] = telemetry["attempt_id"]

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {"a": "a", "b": "b"}


def test_unseeded_context_does_not_reuse_another_contexts_seed():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    compressor._seed_compression_telemetry(
        attempt_id="seeded-attempt",
        session_id="shared-session",
        trigger_source="auto",
    )
    result = {}

    def run_unseeded():
        result.update(
            compressor._begin_compression_telemetry(current_tokens=50_000)
        )

    thread = threading.Thread(target=run_unseeded)
    thread.start()
    thread.join()

    assert result["attempt_id"] != "seeded-attempt"


def test_compressor_cannot_read_another_compressors_context_telemetry():
    first = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    second = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    first._begin_compression_telemetry(current_tokens=50_000, attempt_id="first")

    assert second._current_compression_telemetry() is None


def test_terminal_telemetry_uses_explicit_owning_attempt_id(caplog):
    telemetry = {
        "attempt_id": "attempt-a",
        "aux_call_count": 1,
    }
    agent = _agent(None)
    agent._compression_attempt_id = "attempt-b"
    agent.context_compressor._last_compression_telemetry = telemetry
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        with patch(
            "agent.conversation_compression.time.monotonic",
            return_value=101.0,
        ):
            _emit_compression_attempt_telemetry(
                agent,
                started_at=100.0,
                commit_status="committed",
                split_status="committed",
                attempt_id="attempt-a",
            )

    assert '"attempt_id":"attempt-a"' in caplog.text


def test_compression_result_telemetry_records_token_reduction():
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        quiet_mode=True,
    )
    telemetry = compressor._begin_compression_telemetry(current_tokens=50_000)

    compressor._record_compression_result(
        pre_message_tokens=40_000,
        post_message_tokens=8_000,
    )

    assert telemetry["pre_message_tokens"] == 40_000
    assert telemetry["post_message_tokens"] == 8_000
    assert telemetry["message_tokens_reclaimed"] == 32_000
