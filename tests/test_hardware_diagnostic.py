from types import SimpleNamespace

from scripts.au_doctor import diagnose


class Tensor:
    def __add__(self, _):
        return self

    def sum(self):
        return self

    def item(self):
        return 8


def fake_torch(cuda=False, mps=False, fail_gpu=False):
    def ones(_, device):
        if fail_gpu and device != "cpu":
            raise RuntimeError("device unavailable in this process")
        return Tensor()

    return SimpleNamespace(
        __version__="test", ones=ones,
        cuda=SimpleNamespace(is_available=lambda: cuda,
                             get_device_properties=lambda _: SimpleNamespace(name="test GPU", total_memory=8*1024**3)),
        backends=SimpleNamespace(mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: mps)),
    )


def test_cpu_remains_usable_without_accelerator():
    assert diagnose(fake_torch())["recommended_device"] == "cpu"


def test_mac_selects_mps_only_after_real_tensor_probe():
    assert diagnose(fake_torch(mps=True))["recommended_device"] == "mps"
    report = diagnose(fake_torch(mps=True, fail_gpu=True))
    assert report["recommended_device"] == "cpu"
    assert report["device_checks"][0]["ok"] is False


def test_cuda_reports_actual_memory_for_batch_sizing():
    report = diagnose(fake_torch(cuda=True))
    assert report["recommended_device"] == "0"
    assert report["device_checks"][0]["vram_gib"] == 8.0
