---
type: runbook
owner: codex
updated: 2026-11-12
---

# Test on the M5 Mac and NVIDIA laptop

Djuro's development machine is an Apple M5 MacBook Pro, 16 GB RAM, macOS 27.0.1.
The planned deployment machine is a user-reported Precision 7760 with Xeon and
RTX 3080. Its OS, driver and GPU memory have not been inspected.

## Install and inspect

Use a native arm64 Python 3.12 environment on the Mac. Apple's setup guidance
also lists Xcode command-line tools; install them from a normal Terminal if
needed. On the NVIDIA machine, install a CUDA-enabled PyTorch build using the
[official selector](https://pytorch.org/get-started/locally/); a CPU-only build
will not use the RTX GPU. Then install the repository dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/au_doctor.py
python -m pytest -q
```

On Windows PowerShell activate with `.\.venv\Scripts\Activate.ps1` instead.
`au_doctor.py` reports MPS/CUDA availability, tests tensor computation and reports
actual NVIDIA VRAM when usable. It returns a recommended `--device` value.

Observed here: MPS is built into torch but unavailable to this process; CPU
tensor computation and model inference work. This does not establish that the
physical M5 GPU is unsupported. Run the diagnostic in a normal Terminal before
deciding whether `--device mps` can be used. See
[Apple's MPS verification](https://developer.apple.com/metal/pytorch/) and
[PyTorch MPS notes](https://docs.pytorch.org/docs/2.15/notes/mps.html).

## Prepare the species model once while connected

Use the species' common names; the lion example is a smoke-test prompt, not a
site configuration. Keep the class order identical at preparation and runtime.

```bash
python -m pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
python scripts/au_prepare_model.py --weights yoloe-26s-seg.pt --classes "lion" \
  --out models/yoloe-26s-lion.pt --boxes-only --allow-download
```

Preparation may download the checkpoint and a text encoder. The resulting local
`.pt` stores the prompts; its `.json` sidecar records the version and checksum.
Copy those prepared files to an offline machine. Runtime does not need CLIP or
the text encoder. Use the same checked Ultralytics version on both machines.

## First real clip on the Mac

Start with CPU, batch 1, 150 sampled frames, then compare MPS if the diagnostic
reports it usable. The Mac's 16 GB is shared system/GPU memory; start with the
small or nano model and one process. Keep private footage under ignored `data/`.

```bash
python scripts/au_run.py --video data/raw_videos/test.mp4 --classes "lion" \
  --open-vocab-weights models/yoloe-26s-lion.pt --device cpu \
  --target-fps 5 --max-frames 150 --batch-size 1 --profile \
  --db data/mac-test.db --report-json data/runs/mac-cpu.json
```

Repeat with `--device mps` and a different report path if available. If an MPS
operation fails, retain the error for T1 and use CPU for the next smoke test.
Do not infer GPU speed from the CPU figures in [[pipeline-review-2026-11-12]].

## RTX 3080 comparison

Repeat the same clip, prompts, image size and frame sampling with `--device 0`.
Compare batch sizes 1 and 4 first; increase only after checking actual memory
use and accuracy. Use separate report paths. Model training and larger camera
loads belong on this machine once T1/T2/T3 validation is available. Live sources
must use batch 1 to avoid adding capture latency.

Judge real footage by missed animals, false detections, identity switches and
pacing precision/recall as well as FPS. The synthetic benchmark proves none of
those accuracy metrics. Keep T1 open until its complete acceptance checks pass.
