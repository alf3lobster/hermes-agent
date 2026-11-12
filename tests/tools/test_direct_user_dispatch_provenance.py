import pytest
from types import SimpleNamespace

import model_tools
from agent import tool_executor
from agent.agent_runtime_helpers import invoke_tool
from tools import approval_context


@pytest.fixture(autouse=True)
def _clear_direct_user_provenance_ledgers():
    with model_tools._DIRECT_USER_DISPATCH_LOCK:
        model_tools._DIRECT_USER_DISPATCH_ISSUED.clear()
        model_tools._DIRECT_USER_INSTRUCTION_USED.clear()
    yield
    with model_tools._DIRECT_USER_DISPATCH_LOCK:
        model_tools._DIRECT_USER_DISPATCH_ISSUED.clear()
        model_tools._DIRECT_USER_INSTRUCTION_USED.clear()


def test_direct_handle_function_call_cannot_mint_user_provenance(monkeypatch):
    with pytest.raises(PermissionError, match="executor capability"):
        with model_tools._bind_executor_direct_user_task("invented", object()):
            pass

    captured = []

    def dispatch(name, args, **kwargs):
        captured.append(kwargs.get("dispatch_provenance"))
        return "blocked"

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    session_key_token = approval_context.set_current_session_key("gateway-session")
    try:
        model_tools.handle_function_call(
            "piperis_calendar_write",
            {"operation": "delete"},
            session_id="invented-session",
            turn_id="invented-turn",
            tool_call_id="invented-call",
            user_task="Please delete Private block",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
    finally:
        approval_context.reset_current_session_key(session_key_token)

    assert captured == [None]


def test_executor_supplies_trusted_telegram_continuation_context(monkeypatch):
    consumed = []
    runtime_identity = {
        "session_id": "session-1",
        "session_key": "agent:alfred:telegram:dm:6282086430:22786",
        "source": "telegram",
        "user_id": "6282086430",
        "chat_id": "6282086430",
        "chat_type": "dm",
        "thread_id": "22786",
        "platform_message_id": "23675",
    }

    def dispatch(name, args, **kwargs):
        consumed.append(model_tools.consume_runtime_continuation_context(kwargs["continuation_context"]))
        return "ok"

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    session_key_token = approval_context.set_current_session_key(runtime_identity["session_key"])
    try:
        with model_tools._bind_executor_direct_user_task(
            "Continue the exact request",
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
            runtime_context=runtime_identity,
        ):
            result = model_tools._execute_tool(
                "piperis_calendar_write",
                {"operation": "update", "authority_id": "auth-1"},
                {"operation": "update", "authority_id": "auth-1"},
                model_tools._CallIds(
                    task_id="task-1", session_id="session-1",
                    tool_call_id="call-1", turn_id="turn-1",
                ),
                user_task="Continue the exact request",
                enabled_tools=None,
                skip_tool_execution_middleware=True,
            )
    finally:
        approval_context.reset_current_session_key(session_key_token)

    assert result == "ok"
    assert consumed == [runtime_identity]


def test_direct_caller_cannot_supply_trusted_continuation_context(monkeypatch):
    captured = []
    monkeypatch.setattr(
        model_tools.registry, "dispatch",
        lambda name, args, **kwargs: captured.append(kwargs.get("continuation_context")) or "blocked",
    )
    model_tools.handle_function_call(
        "piperis_calendar_write",
        {"operation": "update", "authority_id": "auth-1"},
        session_id="invented-session", turn_id="invented-turn", tool_call_id="invented-call",
        user_task="Continue", skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )
    assert captured == [None]


def test_execute_tool_mints_dispatch_provenance_only_inside_current_runtime(monkeypatch):
    consumed = []

    def dispatch(name, args, **kwargs):
        consumed.append(model_tools.consume_direct_user_dispatch_provenance(
            kwargs["dispatch_provenance"],
            expected_tool_name="piperis_calendar_write",
            expected_args={"operation": "delete"},
        ))
        return "ok"

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    session_key_token = approval_context.set_current_session_key("gateway-session")
    try:
        with model_tools._bind_executor_direct_user_task(
            "Please delete Private block",
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            result = model_tools._execute_tool(
                "piperis_calendar_write",
                {"operation": "delete"},
                {"operation": "delete"},
                model_tools._CallIds(
                    task_id="task-1",
                    session_id="session-1",
                    tool_call_id="call-1",
                    turn_id="turn-1",
                ),
                user_task="Please delete Private block",
                enabled_tools=None,
                skip_tool_execution_middleware=True,
            )
    finally:
        approval_context.reset_current_session_key(session_key_token)

    assert result == "ok"
    assert consumed == [("Please delete Private block", "session-1", "turn-1", "call-1")]


def test_execute_tool_mints_create_dispatch_provenance_inside_current_runtime(monkeypatch):
    consumed = []
    request = {"operation": "create", "subject": "Private block"}

    def dispatch(name, args, **kwargs):
        consumed.append(model_tools.consume_direct_user_dispatch_provenance(
            kwargs["dispatch_provenance"],
            expected_tool_name="piperis_calendar_write",
            expected_args=request,
        ))
        return "ok"

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    session_key_token = approval_context.set_current_session_key("gateway-session")
    try:
        with model_tools._bind_executor_direct_user_task(
            'Add calendar event titled "Private block"',
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            result = model_tools._execute_tool(
                "piperis_calendar_write",
                request,
                request,
                model_tools._CallIds(
                    task_id="task-1",
                    session_id="session-1",
                    tool_call_id="call-1",
                    turn_id="turn-1",
                ),
                user_task='Add calendar event titled "Private block"',
                enabled_tools=None,
                skip_tool_execution_middleware=True,
            )
    finally:
        approval_context.reset_current_session_key(session_key_token)

    assert result == "ok"
    assert consumed == [('Add calendar event titled "Private block"', "session-1", "turn-1", "call-1")]


def test_execute_tool_mints_update_dispatch_provenance_inside_current_runtime(monkeypatch):
    consumed = []
    request = {"operation": "update", "event_id": "event-1300", "new_subject": "new title"}

    def dispatch(name, args, **kwargs):
        consumed.append(model_tools.consume_direct_user_dispatch_provenance(
            kwargs["dispatch_provenance"],
            expected_tool_name="piperis_calendar_write",
            expected_args=request,
        ))
        return "ok"

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    session_key_token = approval_context.set_current_session_key("gateway-session")
    try:
        with model_tools._bind_executor_direct_user_task(
            'Rename tomorrow\'s 1300 event to "new title"',
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            result = model_tools._execute_tool(
                "piperis_calendar_write",
                request,
                request,
                model_tools._CallIds(
                    task_id="task-1",
                    session_id="session-1",
                    tool_call_id="call-1",
                    turn_id="turn-1",
                ),
                user_task='Rename tomorrow\'s 1300 event to "new title"',
                enabled_tools=None,
                skip_tool_execution_middleware=True,
            )
    finally:
        approval_context.reset_current_session_key(session_key_token)

    assert result == "ok"
    assert consumed == [('Rename tomorrow\'s 1300 event to "new title"', "session-1", "turn-1", "call-1")]


def test_concurrent_invoke_tool_mints_from_transcript_owned_user_row(monkeypatch):
    consumed = []

    def dispatch(name, args, **kwargs):
        consumed.append(model_tools.consume_direct_user_dispatch_provenance(
            kwargs["dispatch_provenance"],
            expected_tool_name="piperis_calendar_write",
            expected_args={"operation": "delete"},
        ))
        return "ok"

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    agent = SimpleNamespace(
        session_id="session-1",
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
        _persist_user_message_idx=0,
        _memory_manager=None,
        valid_tool_names={"piperis_calendar_write"},
        enabled_toolsets=None,
        disabled_toolsets=None,
    )
    messages = [
        {"role": "user", "content": "Please delete Private block"},
        {"role": "assistant", "content": None},
    ]
    session_key_token = approval_context.set_current_session_key("gateway-session")
    observability_tokens = approval_context.set_current_observability_context(
        session_id="session-1", turn_id="turn-1", tool_call_id="call-1"
    )
    try:
        result = invoke_tool(
            agent,
            "piperis_calendar_write",
            {"operation": "delete"},
            "task-1",
            "call-1",
            messages=messages,
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
    finally:
        approval_context.reset_current_observability_context(observability_tokens)
        approval_context.reset_current_session_key(session_key_token)

    assert result == "ok"
    assert consumed == [("Please delete Private block", "session-1", "turn-1", "call-1")]


def test_same_instruction_cannot_issue_two_delete_authorities():
    session_key_token = approval_context.set_current_session_key("gateway-session")
    first_tokens = approval_context.set_current_observability_context(
        session_id="session-1", turn_id="turn-1", tool_call_id="call-1"
    )
    try:
        with model_tools._bind_executor_direct_user_task(
            "Please delete Private block",
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            model_tools._issue_direct_user_dispatch_provenance(
                tool_name="piperis_calendar_write",
                user_task="Please delete Private block",
                session_id="session-1",
                turn_id="turn-1",
                tool_call_id="call-1",
                request_args={"operation": "delete", "event_id": "event-1"},
            )
    finally:
        approval_context.reset_current_observability_context(first_tokens)

    second_tokens = approval_context.set_current_observability_context(
        session_id="session-1", turn_id="turn-1", tool_call_id="call-2"
    )
    try:
        with model_tools._bind_executor_direct_user_task(
            "Please delete Private block",
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            with pytest.raises(PermissionError, match="already used"):
                model_tools._issue_direct_user_dispatch_provenance(
                    tool_name="piperis_calendar_write",
                    user_task="Please delete Private block",
                    session_id="session-1",
                    turn_id="turn-1",
                    tool_call_id="call-2",
                    request_args={"operation": "delete", "event_id": "event-2"},
                )
    finally:
        approval_context.reset_current_observability_context(second_tokens)
        approval_context.reset_current_session_key(session_key_token)


def test_dispatch_provenance_is_opaque_bound_and_one_use():
    session_key_token = approval_context.set_current_session_key("gateway-session")
    observability_tokens = approval_context.set_current_observability_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
    )
    try:
        with model_tools._bind_executor_direct_user_task(
            "Please delete Private block",
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            provenance = model_tools._issue_direct_user_dispatch_provenance(
                tool_name="piperis_calendar_write",
                user_task="Please delete Private block",
                session_id="session-1",
                turn_id="turn-1",
                tool_call_id="call-1",
                request_args={"operation": "delete"},
            )
        with pytest.raises(PermissionError, match="exact request"):
            model_tools.consume_direct_user_dispatch_provenance(
                provenance,
                expected_tool_name="piperis_calendar_write",
                expected_args={"operation": "delete", "event_id": "substituted"},
            )
        assert model_tools.consume_direct_user_dispatch_provenance(
            provenance,
            expected_tool_name="piperis_calendar_write",
            expected_args={"operation": "delete"},
        ) == (
            "Please delete Private block",
            "session-1",
            "turn-1",
            "call-1",
        )
        with pytest.raises(PermissionError, match="already consumed"):
            model_tools.consume_direct_user_dispatch_provenance(
                provenance,
                expected_tool_name="piperis_calendar_write",
                expected_args={"operation": "delete"},
            )
    finally:
        approval_context.reset_current_observability_context(observability_tokens)
        approval_context.reset_current_session_key(session_key_token)


def test_dispatch_provenance_rejects_forgery_and_cross_session():
    with pytest.raises(PermissionError, match="invalid"):
        model_tools.consume_direct_user_dispatch_provenance(
            object(),
            expected_tool_name="piperis_calendar_write",
            expected_args={"operation": "delete"},
        )

    session_key_token = approval_context.set_current_session_key("gateway-session")
    issue_tokens = approval_context.set_current_observability_context(
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
    )
    try:
        with model_tools._bind_executor_direct_user_task(
            "Please delete Private block",
            tool_executor._DIRECT_USER_TASK_CAPABILITY,
        ):
            provenance = model_tools._issue_direct_user_dispatch_provenance(
                tool_name="piperis_calendar_write",
                user_task="Please delete Private block",
                session_id="session-1",
                turn_id="turn-1",
                tool_call_id="call-1",
                request_args={"operation": "delete"},
            )
    finally:
        approval_context.reset_current_observability_context(issue_tokens)

    consume_tokens = approval_context.set_current_observability_context(
        session_id="session-2",
        turn_id="turn-1",
        tool_call_id="call-1",
    )
    try:
        with pytest.raises(PermissionError, match="runtime"):
            model_tools.consume_direct_user_dispatch_provenance(
                provenance,
                expected_tool_name="piperis_calendar_write",
                expected_args={"operation": "delete"},
            )
    finally:
        approval_context.reset_current_observability_context(consume_tokens)
        approval_context.reset_current_session_key(session_key_token)
