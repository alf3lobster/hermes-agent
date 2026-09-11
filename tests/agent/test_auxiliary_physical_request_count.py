"""Actual provider requests are exposed to auxiliary callers."""

from types import SimpleNamespace
from unittest.mock import patch

from agent import auxiliary_client as aux
from agent.auxiliary_client import _relay_auxiliary_call, _relay_sync_completion


def test_route_info_counts_every_physical_provider_request():
    route_info = {}

    @_relay_auxiliary_call
    def run(*, task, route_info):
        request = {"model": "test-model", "messages": []}
        _relay_sync_completion(object(), request, create=lambda _: object())
        _relay_sync_completion(object(), request, create=lambda _: object())

    run(task="compression", route_info=route_info)

    assert route_info["physical_request_count"] == 2


def test_call_llm_counts_transient_retry_at_production_boundary():
    attempts = 0

    def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("brief transport failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    client = SimpleNamespace(
        base_url="https://example.invalid",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    route_info = {}
    with (
        patch.object(
            aux,
            "_resolve_task_provider_model",
            return_value=("custom", "test-model", None, "key", "chat_completions"),
        ),
        patch.object(aux, "_get_cached_client", return_value=(client, "test-model")),
        patch.object(aux, "_effective_provider_for_client", return_value="custom"),
        patch.object(aux, "_effective_aux_timeout", return_value=5.0),
        patch.object(aux, "_transient_retry_count", return_value=1),
        patch.object(aux.time, "sleep", return_value=None),
    ):
        response = aux.call_llm(
            task="compression",
            provider="custom",
            messages=[{"role": "user", "content": "summarise"}],
            route_info=route_info,
        )

    assert response.choices[0].message.content == "ok"
    assert attempts == 2
    assert route_info["physical_request_count"] == 2


def test_pre_dispatch_failure_counts_zero_physical_requests():
    route_info = {}
    with patch.object(
        aux,
        "_resolve_task_provider_model",
        side_effect=RuntimeError("resolution failed before provider dispatch"),
    ):
        try:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarise"}],
                route_info=route_info,
            )
        except RuntimeError:
            pass

    assert route_info.get("physical_request_count", 0) == 0