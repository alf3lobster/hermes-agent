from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "install-e2e-run.yml"


def test_install_e2e_inputs_are_data_not_shell_source():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    bindings = (
        ("E2E_INSTALL_METHOD", "install-method"),
        ("E2E_UPDATE_METHOD", "update-method"),
        ("E2E_INSTALL_REF", "install-ref"),
        ("E2E_UPDATE_REF", "update-ref"),
        ("E2E_LEG_ID", "leg-id"),
    )
    for env_name, input_name in bindings:
        assert f"{env_name}: ${{{{ inputs.{input_name} }}}}" in workflow

    assert '--install-method "$E2E_INSTALL_METHOD"' in workflow
    assert '--update-method "$E2E_UPDATE_METHOD"' in workflow
    assert '--install-ref "$E2E_INSTALL_REF"' in workflow
    assert '--update-ref "$E2E_UPDATE_REF"' in workflow
    assert 'echo "name=install-e2e-logs-${E2E_LEG_ID}"' in workflow

    assert "--install-method '${{ inputs.install-method }}'" not in workflow
    assert "--update-method '${{ inputs.update-method }}'" not in workflow
    assert "--install-ref '${{ inputs.install-ref }}'" not in workflow
    assert "--update-ref '${{ inputs.update-ref }}'" not in workflow
    assert 'echo "name=install-e2e-logs-${{ inputs.leg-id }}"' not in workflow
