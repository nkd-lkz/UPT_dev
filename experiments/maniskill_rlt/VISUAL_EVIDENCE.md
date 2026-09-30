# Visual Evidence Diagnostic Prototype

This branch measures how controlled image occlusions affect separate output paths, helping distinguish VLA reference-action failures from Stage 2 actor failures. It provides an offline interface and CPU tests. Real-checkpoint integration, attention extraction and control benefits remain untested.

## Use the Interface

In `toolkits/rlt/visual_evidence.py`, `ImageRegion` defines a half-open rectangle in a named camera. `diagnose_regions` accepts floating `[B,C,H,W]` images, regions and a deterministic callback. Capture fixed language, proprioception and action noise in that callback; freeze weights and use eval mode. Return separate `vla_action`, `rl_token` and `actor_action` outputs rather than combining their units into one map.

This toy example demonstrates the contract, not a VLA adapter:

```python
import torch
from toolkits.rlt.visual_evidence import ImageRegion, diagnose_regions

images = {"main": torch.ones(1, 3, 8, 8)}
def predict(images):
    return {"example_feature": images["main"].mean((2, 3))}

report = diagnose_regions(
    images, [ImageRegion("main", 0, 0, 4, 4)], predict,
    replacement="zero",
)
```

Outputs contain per-path, per-sample RMS changes. Same-shape demonstration targets additionally enable signed MSE increases: negative values mean the perturbation improved target agreement and must not be clipped away. `mean` fills with each camera channel's spatial mean; `zero` depends on normalization and does not always mean black. Both interventions can be out of distribution.

Two unmodified-image calls must agree exactly before measurement, separating sampling noise from sensitivity. This runtime check cannot establish that a callback has no hidden state. N regions require N+2 predictions, with a 64-region limit. The helper does not change global RNG, weights or device placement; callers remain responsible for callback side effects. Start with recorded frames, outside the real-time control loop.

## Relation to Prior Work and Next Checks

[ActGaze](https://arxiv.org/html/2609.28955v1) already derives action-grounded gaze supervision from visual interventions; [VLA-Trace](https://arxiv.org/html/2605.30117v2) separates representation and behavior across model stages. This helper is not a new attention algorithm or causal explanation.

For a real adapter, first verify unchanged-image parity with original inference under fixed noise. Compare the gripper–peg tip–hole relation against equal-area background regions and another camera. The API neither checks action feasibility nor chooses regions automatically; it introduces no future outcome or simulator truth into deployment inputs. Next add data adapters and versioned offline outputs. Study history-conditioned evidence selection only after diagnosing repeatable failure mechanisms.

CPU check: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit_tests/test_visual_evidence.py -q`. Eight tests pass; GPU and robot evaluations have not run.
