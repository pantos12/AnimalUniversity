---
type: adr
owner: codex
status: proposed
date: 2026-11-12
---

# ADR-004: Prepare models once; validate with independent recording groups

## Context

The initial runtime could download model/text assets unexpectedly. Its
auto-labeler assigned neighboring frames to different splits, replaced dataset
class definitions on append, and could overwrite earlier pseudo-labels.

## Decision

Runtime uses local checkpoints and cached class prompts. A separate explicit
connected setup step prepares prompts, optionally using a boxes-only YOLOE
architecture, and records a checksum. No larger or smaller model becomes an
accuracy default on the basis of synthetic throughput alone.

Archived batching is bounded and optional. Tracker, behavior and sink calls
remain sequential in source order. Live inference retains batch size 1.

Auto-label splits belong to recording groups, with immutable class ordering,
stable recording IDs and an append manifest. Clips from a related camera/day
should share a group. Empty detections are not verified background labels.
Incomplete datasets and datasets without independent train/val recordings are
blocked before auto-label training starts.

## Consequences

Connected setup must happen before offline deployment. Existing auto-label
datasets need a new output directory; migration must validate old class order
and split provenance rather than guessing. Single clips no longer yield
artificial train/val metrics. Actual animal accuracy, CUDA/MPS performance and
T1 acceptance remain unverified. See [[pipeline-review-2026-11-12]].
