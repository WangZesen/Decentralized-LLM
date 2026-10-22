import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from decentralized_llm.config import Config, save_config
from decentralized_llm.data.cache import TokenCache, training_boundaries
from decentralized_llm.launch.slurm import SubmissionRejected
from decentralized_llm.runtime import atomic_json
from scripts import precision_campaign as campaign


def fixture_campaign(tmp_path: Path, tiny_config: Config) -> tuple[dict, dict[str, Config]]:
    raw = tiny_config.model_dump(mode="json")
    raw["runtime"].update(device="cuda", training_backend="packed", amp=True, deterministic=False)
    raw["distributed"].update(mode="decentralized", nodes=2, gpus_per_node=4, topology="one_peer_ring", scheme="awc")
    raw["distributed"].pop("error_feedback_gamma")
    raw["distributed"]["error_feedback"] = True
    raw["training"].update(batch_tokens=64, num_accumulation=1)
    raw["launcher"] = {"kind": "slurm", "partition": "gpu"}
    winner = tmp_path / "winner.yaml"
    winner.write_text(yaml.safe_dump(raw))
    configs = campaign.derive_configs(winner, tmp_path)
    manifest = {"source_sha256": "source", "source": str(tmp_path), "runs": []}
    for key, config in configs.items():
        config.runtime.output_dir.mkdir(parents=True)
        path = config.runtime.output_dir / "resolved.yaml"
        save_config(config, path)
        manifest["runs"].append(
            {
                "key": key,
                "mode": key.split("/")[0],
                "seed": config.runtime.seed,
                "run": str(config.runtime.output_dir),
                "config": str(path),
                "status": "prepared",
            }
        )
    return manifest, configs


def test_paired_campaign_preserves_winner_and_inputs(tmp_path: Path, tiny_config: Config, cache: TokenCache) -> None:
    _, configs = fixture_campaign(tmp_path, tiny_config)
    assert len(configs) == 6
    evidence = campaign.paired_inputs(configs)
    assert [r["seed"] for r in evidence] == [42, 43, 44]
    for config in configs.values():
        assert config.optimizer == tiny_config.optimizer
        assert config.distributed.error_feedback_gamma == 1.0
        assert config.distributed.world_size == 8 and config.physical_gpu_count == 1
        assert config.runtime.amp is False and config.runtime.compile is False
        assert config.evaluation.fp32_precision == "ieee" and config.runtime.sdpa_backend == "math"


def test_report_marks_missing_failed_and_partial_groups(tmp_path: Path, tiny_config: Config) -> None:
    manifest, configs = fixture_campaign(tmp_path, tiny_config)
    for mode, loss in (("tf32", 3.6), ("fp32", 3.5)):
        cfg = configs[f"{mode}/seed-42"]
        atomic_json(
            cfg.runtime.output_dir / "result.json",
            {
                "cursor": training_boundaries(cfg, cfg.model.parameter_count)[-1],
                "completed_epochs": cfg.training.epoch_count,
                "validation": {"loss": loss, "perplexity": 33.0, "consensus_error": 1.0, "split": "full"},
            },
        )
    atomic_json(configs["tf32/seed-43"].runtime.output_dir / "failure-rank-00000.json", {"kind": "nonfinite_update"})
    result = campaign.summarize(manifest)
    assert result["groups"]["tf32"]["mean_loss"] is None
    assert result["paired_differences"][0]["tf32_minus_fp32_loss"] == pytest.approx(0.1)
    assert [r["status"] for r in result["runs"]] == ["complete", "failed", "missing", "complete", "missing", "missing"]


def test_submit_requires_smoke_and_does_not_duplicate(
    tmp_path: Path, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = fixture_campaign(tmp_path, tiny_config)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda _: None)
    with pytest.raises(FileNotFoundError):
        campaign.submit(tmp_path, manifest)
    (tmp_path / "smoke").mkdir()
    atomic_json(tmp_path / "smoke/result.json", {"passed": True, "source_sha256": "source"})
    calls = []

    def launch(config: Config, **kwargs: Any):
        calls.append(config)
        destination = kwargs["submission_dir"]
        destination.mkdir()
        receipt = {"job_id": str(len(calls))}
        atomic_json(destination / "receipt.json", receipt)
        return receipt

    monkeypatch.setattr(campaign, "launch", launch)
    first = campaign.submit(tmp_path, manifest)
    second = campaign.submit(tmp_path, json.loads((tmp_path / "campaign.json").read_text()))
    assert first == second and len(calls) == 6
    manifest["runs"][0].pop("job_id")
    manifest["runs"][0]["status"] = "submitting"
    # A durable launcher receipt recovers a crash before the campaign manifest was updated.
    assert campaign.submit(tmp_path, manifest) == first
    assert len(calls) == 6


@pytest.mark.parametrize("rejected", [False, True])
def test_smoke_retries_only_explicit_scheduler_rejections(
    tmp_path: Path, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, rejected: bool
) -> None:
    manifest, _ = fixture_campaign(tmp_path, tiny_config)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda _: None)
    calls = []

    def submit(arguments: list[str], script: Path) -> dict:
        calls.append(script)
        if len(calls) == 1:
            if rejected:
                raise SubmissionRejected("scheduler rejected smoke job")
            raise OSError("submission outcome unknown")
        return {"job_id": "123", "cluster": None}

    monkeypatch.setattr(campaign, "submit_batch", submit)
    with pytest.raises(SubmissionRejected if rejected else OSError):
        campaign.smoke_submit(tmp_path, manifest)
    saved = json.loads((tmp_path / "campaign.json").read_text())
    assert saved["smoke"]["status"] == ("rejected" if rejected else "submitting")
    result = campaign.smoke_submit(tmp_path, saved)
    assert result["status"] == ("submitted" if rejected else "submitting")
    assert len(calls) == (2 if rejected else 1)
    persisted = json.loads((tmp_path / "campaign.json").read_text())
    assert campaign.smoke_submit(tmp_path, persisted) == result
    assert len(calls) == (2 if rejected else 1)
