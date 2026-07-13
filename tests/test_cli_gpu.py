from click.testing import CliRunner

from affine.cli.main import cli


def test_gpu_replace_endpoint_forwards_options(monkeypatch):
    captured = {}

    async def fake_replace_endpoint_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "affine.src.scheduler.gpu_autoscaler.replace_endpoint_command",
        fake_replace_endpoint_command,
    )

    result = CliRunner().invoke(
        cli,
        [
            "gpu",
            "replace-endpoint",
            "--old",
            "lium-b200-1",
            "--new-slot",
            "lium-b200-2",
            "--keep-old",
            "--dry-run",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "old_endpoint_name": "lium-b200-1",
        "new_slot_name": "lium-b200-2",
        "keep_old": True,
        "dry_run": True,
        "yes": True,
    }


def test_gpu_remove_endpoint_forwards_options(monkeypatch):
    captured = {}

    async def fake_remove_endpoint_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "affine.src.scheduler.gpu_autoscaler.remove_endpoint_command",
        fake_remove_endpoint_command,
    )

    result = CliRunner().invoke(
        cli,
        [
            "gpu",
            "remove-endpoint",
            "--name",
            "lium-b200-temp-2",
            "--dry-run",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "endpoint_name": "lium-b200-temp-2",
        "keep_slot": False,
        "dry_run": True,
        "yes": True,
    }
