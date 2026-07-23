"""Smoke tests for `munbench run`: stage ordering, skip flags, and stop-on-failure.
All sub-stages are mocked - no generation/judging/network happens here."""

import pytest
import typer

import munbench.cli as cli_mod


def _write_config(tmp_path):
    config_path = tmp_path / "munbench.yaml"
    config_path.write_text("models:\n  - fake-model\n", encoding="utf-8")
    return config_path


def test_run_executes_all_stages_in_order(monkeypatch, tmp_path):
    calls: list[str] = []

    def make_stage(name):
        def _stage(*args, **kwargs):
            calls.append(name)

        return _stage

    monkeypatch.setattr(cli_mod, "validate_data", make_stage("validate-data"))
    monkeypatch.setattr(cli_mod, "generate", make_stage("generate"))
    monkeypatch.setattr(cli_mod, "judge", make_stage("judge"))
    monkeypatch.setattr(cli_mod, "elo", make_stage("elo"))
    monkeypatch.setattr(cli_mod, "report", make_stage("report"))

    config_path = _write_config(tmp_path)
    cli_mod.run(
        track="all", models=None, config=config_path,
        skip_generate=False, skip_rubric=False, skip_pairwise=False, skip_elo=False, skip_report=False,
    )

    assert calls == ["validate-data", "generate", "judge", "judge", "elo", "report"]


def test_run_respects_skip_flags(monkeypatch, tmp_path):
    calls: list[str] = []

    def make_stage(name):
        def _stage(*args, **kwargs):
            calls.append(name)

        return _stage

    monkeypatch.setattr(cli_mod, "validate_data", make_stage("validate-data"))
    monkeypatch.setattr(cli_mod, "generate", make_stage("generate"))
    monkeypatch.setattr(cli_mod, "judge", make_stage("judge"))
    monkeypatch.setattr(cli_mod, "elo", make_stage("elo"))
    monkeypatch.setattr(cli_mod, "report", make_stage("report"))

    config_path = _write_config(tmp_path)
    cli_mod.run(
        track="all", models=None, config=config_path,
        skip_generate=True, skip_rubric=True, skip_pairwise=False, skip_elo=False, skip_report=False,
    )

    # generate skipped entirely; judge runs once (pairwise only, rubric skipped)
    assert calls == ["validate-data", "judge", "elo", "report"]


def test_run_stops_on_first_stage_failure(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_validate_data(config):
        calls.append("validate-data")

    def fake_generate(track, models, config):
        calls.append("generate")
        raise typer.Exit(1)

    def fake_judge(mode, models, config):
        calls.append(f"judge-{mode}")

    monkeypatch.setattr(cli_mod, "validate_data", fake_validate_data)
    monkeypatch.setattr(cli_mod, "generate", fake_generate)
    monkeypatch.setattr(cli_mod, "judge", fake_judge)
    monkeypatch.setattr(cli_mod, "elo", lambda config: calls.append("elo"))
    monkeypatch.setattr(cli_mod, "report", lambda models, config: calls.append("report"))

    config_path = _write_config(tmp_path)
    with pytest.raises(typer.Exit):
        cli_mod.run(
            track="all", models=None, config=config_path,
            skip_generate=False, skip_rubric=False, skip_pairwise=False, skip_elo=False, skip_report=False,
        )

    # stopped immediately after "generate" failed - rubric/pairwise/elo/report never ran
    assert calls == ["validate-data", "generate"]


def test_run_requires_models(tmp_path):
    config_path = tmp_path / "munbench.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        cli_mod.run(
            track="all", models=None, config=config_path,
            skip_generate=False, skip_rubric=False, skip_pairwise=False, skip_elo=False, skip_report=False,
        )
