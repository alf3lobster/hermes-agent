"""Direct SIGKILL requires approval; this suite never sends a signal."""
import pytest
from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize("command", [
    "kill -9 12345", "kill -KILL 12345", "kill -SIGKILL 12345",
    "/bin/kill -9 12345", "/usr/bin/kill -KILL 12345",
    "'/bin/kill' '-SIGKILL' 12345", 'k"ill" -"KILL" 12345',
    "kill -s KILL 12345", "kill -s SIGKILL 12345", "kill -s 9 12345",
    "kill --signal KILL 12345", "kill --signal=SIGKILL 12345",
    "kill --signal=9 12345", "command kill -9 12345",
    "env kill -9 12345", "true && kill -9 12345",
    "printf '%s' \"$(kill -9 12345)\"", "kill -9 -- -12345",
])
def test_force_kill_requires_approval(command):
    assert detect_dangerous_command(command)[0], command


@pytest.mark.parametrize("command", [
    "kill 12345", "kill -TERM 12345", "kill -SIGTERM 12345",
    "kill -15 12345", "/bin/kill -0 12345", "kill -s TERM 12345",
    "kill -l", "kill -l 9", "kill -l KILL",
    "printf '%s' 'kill -9 12345'", "echo kill -9 12345",
    "true # kill -9 12345", "kill 12345; echo -9",
])
def test_non_force_and_prose_keep_existing_policy(command):
    assert not detect_dangerous_command(command)[0], command
