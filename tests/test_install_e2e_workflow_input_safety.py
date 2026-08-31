from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "install-e2e-run.yml"


def test_install_e2e_inputs_are_data_not_shell_source():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "E2E_ROUTE: ${{ inputs.route }}" in workflow
    assert "E2E_INSTALL_REF: ${{ inputs.install-ref }}" in workflow
    assert 'case "$E2E_ROUTE" in' in workflow
    assert '--install-ref "$E2E_INSTALL_REF"' in workflow
    assert "case '${{ inputs.route }}' in" not in workflow
    assert "--install-ref '${{ inputs.install-ref }}'" not in workflow
