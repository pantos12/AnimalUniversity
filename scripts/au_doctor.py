"""Report the actual PyTorch devices and test a small tensor operation."""
from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version


def diagnose(torch) -> dict:
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "torch": torch.__version__,
        "cuda_available": cuda,
        "mps_built": bool(mps and mps.is_built()),
        "mps_available": bool(mps and mps.is_available()),
        "device_checks": [], "recommended_device": None,
    }
    try:
        report["ultralytics"] = version("ultralytics")
    except PackageNotFoundError:
        report["ultralytics"] = None
    candidates = (["cuda:0"] if cuda else []) + (["mps"] if report["mps_available"] else []) + ["cpu"]
    for device in candidates:
        try:
            value = (torch.ones(4, device=device) + 1).sum().item()
            if value != 8:
                raise RuntimeError("Unexpected tensor probe result")
            entry = {"device": device, "ok": True}
            if device.startswith("cuda"):
                props = torch.cuda.get_device_properties(0)
                entry.update(name=props.name, vram_gib=round(props.total_memory / 1024**3, 2))
            report["device_checks"].append(entry)
            report["recommended_device"] = "0" if device.startswith("cuda") else device
            break
        except Exception as exc:
            report["device_checks"].append({"device": device, "ok": False, "error": str(exc)})
    return report


def main() -> int:
    try:
        import torch
    except ImportError:
        print(json.dumps({"error": "PyTorch is not installed in this Python environment"}))
        return 2
    report = diagnose(torch)
    print(json.dumps(report, indent=2))
    return 0 if report["recommended_device"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
