"""GPU pool identities and reservations, without opening a CUDA device."""

from types import SimpleNamespace

import pytest

from decentralized_llm.tuning import local_gpus
from decentralized_llm.tuning.local_gpus import available_gpus, discover_gpus, restore_pool
from decentralized_llm.tuning.spec import Controller, Record

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
GPU_C = "GPU-00000000-0000-0000-0000-000000000003"


@pytest.fixture
def visible_gpus(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    identifiers = [GPU_A, GPU_B, GPU_C]
    monkeypatch.setattr(local_gpus.socket, "gethostname", lambda: "worker-host")
    monkeypatch.setattr(local_gpus.torch.cuda, "device_count", lambda: len(identifiers))
    monkeypatch.setattr(
        local_gpus.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=identifiers[index])
    )
    return identifiers


def test_discovery_respects_visible_ordinals_and_selection_order(
    visible_gpus: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2,7")
    assert discover_gpus() == {"host": "worker-host", "gpu_ids": visible_gpus}
    assert discover_gpus([2, 0]) == {"host": "worker-host", "gpu_ids": [GPU_C, GPU_A]}


def test_discovery_normalizes_pytorch_uuid_payloads(visible_gpus: list[str]) -> None:
    # torch._C._CUuuid.__str__ returns an unprefixed UUID on real CUDA hardware.
    visible_gpus[:] = [identifier.removeprefix("GPU-") for identifier in visible_gpus]
    assert discover_gpus([2, 0])["gpu_ids"] == [GPU_C, GPU_A]


def test_bare_mig_uuid_preserves_explicit_device_identity(
    visible_gpus: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    visible_gpus[0] = GPU_A.removeprefix("GPU-")
    mig = "MIG-" + visible_gpus[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mig)
    assert discover_gpus([0])["gpu_ids"] == [mig]


@pytest.mark.parametrize(
    "mask", ["MIG-unknown", "MIG-00000000-0000-0000-0000-000000000002", GPU_A.replace("GPU-", "MIG-GPU-") + "/1/0"]
)
def test_bare_mig_uuid_never_guesses_from_an_ambiguous_mask(
    visible_gpus: list[str], monkeypatch: pytest.MonkeyPatch, mask: str
) -> None:
    visible_gpus[0] = GPU_A.removeprefix("GPU-")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="cannot resolve bare MIG identity"):
        discover_gpus([0])


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [3], [True], [1.0], ["0"]])
def test_discovery_rejects_invalid_selection(visible_gpus: list[str], indices: list) -> None:
    with pytest.raises(ValueError, match="unique visible CUDA indices"):
        discover_gpus(indices)


def test_discovery_without_visible_gpus(visible_gpus: list[str]) -> None:
    visible_gpus.clear()
    with pytest.raises(ValueError, match="at least one visible CUDA device"):
        discover_gpus()


@pytest.mark.parametrize("identifier", ["", "0", "GPU-short", "GPU-a,b", "GPU-None"])
def test_discovery_never_falls_back_to_an_ordinal(visible_gpus: list[str], identifier: str) -> None:
    visible_gpus[0] = identifier
    with pytest.raises(ValueError, match="complete GPU UUID"):
        discover_gpus([0])


def test_discovery_requires_distinct_device_identities(visible_gpus: list[str]) -> None:
    visible_gpus[1] = visible_gpus[0]
    with pytest.raises(ValueError, match="unique, complete GPU or MIG UUIDs"):
        discover_gpus()


@pytest.mark.parametrize(
    "identifier",
    ["MIG-00000000-0000-0000-0000-000000000001", "MIG-GPU-00000000-0000-0000-0000-000000000001/2/1"],
)
def test_discovery_preserves_mig_identity(visible_gpus: list[str], identifier: str) -> None:
    visible_gpus[0] = identifier
    assert discover_gpus([0])["gpu_ids"] == [identifier]


def test_resume_preserves_pool_across_reordered_visibility(visible_gpus: list[str]) -> None:
    saved = discover_gpus([0, 2])
    visible_gpus.reverse()
    restored = restore_pool(saved, discover_gpus())
    assert restored == {"host": "worker-host", "gpu_ids": [GPU_A, GPU_C]}
    assert restored is not saved and restored["gpu_ids"] is not saved["gpu_ids"]


def test_resume_does_not_bypass_current_visibility(visible_gpus: list[str]) -> None:
    saved = discover_gpus()
    visible_gpus.remove(GPU_B)
    with pytest.raises(ValueError, match="restore CUDA_VISIBLE_DEVICES"):
        restore_pool(saved, discover_gpus())


def test_resume_rejects_another_host(visible_gpus: list[str]) -> None:
    saved = {**discover_gpus(), "host": "original-host"}
    with pytest.raises(ValueError, match="resume it on that host"):
        restore_pool(saved, discover_gpus())


def test_reservations_preserve_available_order_and_old_revision_work(visible_gpus: list[str]) -> None:
    pool = discover_gpus([0, 1])
    active = [{"host": "worker-host", "local_gpu_ids": [GPU_B, GPU_C]}]
    assert available_gpus(pool, active) == [GPU_A]
    assert available_gpus(pool, []) == [GPU_A, GPU_B]


@pytest.mark.parametrize("attempt", [{}, {"local_gpu_ids": []}])
def test_unassigned_live_attempt_reserves_entire_pool(visible_gpus: list[str], attempt: Record) -> None:
    assert available_gpus(discover_gpus(), [attempt]) == []


def test_reservations_reject_another_host(visible_gpus: list[str]) -> None:
    with pytest.raises(ValueError, match="different host"):
        available_gpus(discover_gpus(), [{"host": "other-host", "local_gpu_ids": [GPU_A]}])


def test_overlapping_live_assignments_fail_closed(visible_gpus: list[str]) -> None:
    active = [{"local_gpu_ids": [GPU_A, GPU_C]}, {"local_gpu_ids": [GPU_C]}]
    with pytest.raises(ValueError, match="assignments overlap"):
        available_gpus(discover_gpus([0, 1]), active)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [True], [1.0], ["0"]])
def test_controller_rejects_invalid_local_gpu_lists(indices: list) -> None:
    with pytest.raises(ValueError):
        Controller.model_validate({"local_gpus": indices})


def test_controller_defaults_to_all_visible_gpus() -> None:
    assert Controller().local_gpus is None
    assert Controller(local_gpus=[2, 0]).local_gpus == [2, 0]
