from types import SimpleNamespace


def test_per_call_provider_and_model_resolve_full_bundle(monkeypatch):
    from tools import delegate_tool

    seen = {}

    def resolve(cfg, parent):
        seen.update(cfg)
        return {
            "model": cfg.get("model"),
            "provider": cfg.get("provider"),
            "base_url": "https://provider.example/v1",
            "api_key": "scoped-key",
            "api_mode": "responses",
            "request_overrides": {"service_tier": "priority"},
            "command": None,
            "args": [],
        }

    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", resolve)
    base = {"model": "configured-model", "provider": "configured-provider", "fallback_chain": ["safe"]}

    routing, creds = delegate_tool._resolve_call_route(
        base, SimpleNamespace(model="parent-model"), model="call-model", provider="call-provider"
    )

    assert routing == {
        "model": "call-model",
        "provider": "call-provider",
        "fallback_chain": ["safe"],
    }
    assert seen == routing
    assert creds["model"] == "call-model"
    assert creds["provider"] == "call-provider"
    assert creds["base_url"] == "https://provider.example/v1"
    assert creds["api_key"] == "scoped-key"
    assert creds["api_mode"] == "responses"
    assert creds["request_overrides"] == {"service_tier": "priority"}


def test_delegation_schema_advertises_string_route_overrides():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    properties = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert properties["model"]["type"] == "string"
    assert properties["provider"]["type"] == "string"


def test_heartbeat_touches_parent_immediately_before_scheduling(monkeypatch):
    from tools import delegate_tool_child_run
    from agent import periodic_scheduler

    events = []
    parent = SimpleNamespace(_touch_activity=lambda description: events.append(("touch", description)))
    child = SimpleNamespace(
        get_activity_summary=lambda: {
            "current_tool": None,
            "api_call_count": 0,
            "max_iterations": 50,
            "last_activity_ts": 1.0,
            "last_activity_desc": "starting",
        }
    )
    handle = SimpleNamespace(cancel=lambda wait=0: None)
    monkeypatch.setattr(periodic_scheduler, "schedule", lambda callback, interval: events.append(("schedule", interval)) or handle)

    heartbeat = delegate_tool_child_run._Heartbeat(child, parent, 0)
    heartbeat.start()

    assert events[0][0] == "touch"
    assert events[1][0] == "schedule"
