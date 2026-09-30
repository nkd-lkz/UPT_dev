# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure visual perturbation sensitivity, not attention or causal importance."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImageRegion:
    """A half-open pixel rectangle in one named camera image."""

    camera: str
    top: int
    left: int
    bottom: int
    right: int


@torch.no_grad()
def diagnose_regions(
    images: Mapping[str, torch.Tensor],
    regions: Sequence[ImageRegion],
    predict: Callable[[dict[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    *,
    replacement: str = "mean",
    targets: Mapping[str, torch.Tensor] | None = None,
) -> dict:
    """Compare named output paths under matched, one-region image perturbations.

    Images are finite float [B,C,H,W] tensors. ``predict`` must be side-effect
    free and deterministic: freeze weights, use eval mode, and reuse the same
    action noise for every call. Capture state and language in its closure. Use
    distinct output keys for VLA actions, RL tokens and actor actions. The helper
    never changes model weights, RNG state, device placement or controller state.

    Each output has a per-sample RMS change and, when a same-shape target is
    supplied, a signed MSE increase. Without targets, sensitivity cannot tell
    whether an action became better or worse. Values across output keys have
    different units and must not be merged into one importance score. At most
    64 regions are allowed; cost is 2 + len(regions) predictions.
    """
    if not images or not 1 <= len(regions) <= 64:
        raise ValueError("Provide images and between 1 and 64 regions")
    if replacement not in ("mean", "zero"):
        raise ValueError("replacement must be mean or zero")
    first_image = next(iter(images.values()))
    if first_image.ndim != 4:
        raise ValueError("Expected finite floating [B,C,H,W] images")
    batch = first_image.shape[0]
    for value in images.values():
        if (
            value.ndim != 4
            or min(value.shape) < 1
            or value.shape[0] != batch
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Expected finite floating [B,C,H,W] images")
    for region in regions:
        if region.camera not in images:
            raise ValueError("Unknown camera")
        height, width = images[region.camera].shape[-2:]
        coords = (region.top, region.left, region.bottom, region.right)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in coords) or not (
            0 <= region.top < region.bottom <= height
            and 0 <= region.left < region.right <= width
        ):
            raise ValueError("Region must be an in-bounds nonempty rectangle")

    def evaluate(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # A callback cannot corrupt subsequent occlusions by mutating its input.
        outputs = predict({key: value.clone() for key, value in inputs.items()})
        if not outputs:
            raise ValueError("Predictor returned no output paths")
        for value in outputs.values():
            if (
                value.ndim < 2
                or value.shape[0] != batch
                or value.numel() == 0
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Expected finite floating [B,...] outputs")
        return {
            key: value.detach().float().cpu().clone() for key, value in outputs.items()
        }

    baseline = evaluate(images)
    repeated = evaluate(images)
    if baseline.keys() != repeated.keys() or any(
        not torch.equal(value, repeated[key]) for key, value in baseline.items()
    ):
        raise ValueError(
            "Predictor is not deterministic; fix action noise and model state"
        )
    target_values = {}
    for key, value in (targets or {}).items():
        if (
            key not in baseline
            or value.shape != baseline[key].shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Targets must be finite and match a named output")
        target_values[key] = value.detach().float().cpu().clone()

    results = []
    for region in regions:
        inputs = dict(images)
        original = images[region.camera]
        changed = original.clone()
        fill = original.mean(dim=(-2, -1), keepdim=True) if replacement == "mean" else 0
        changed[..., region.top : region.bottom, region.left : region.right] = fill
        inputs[region.camera] = changed
        outputs = evaluate(inputs)
        if outputs.keys() != baseline.keys() or any(
            value.shape != baseline[key].shape for key, value in outputs.items()
        ):
            raise ValueError("Output paths and shapes must stay constant")
        paths = {}
        for key, value in outputs.items():
            paths[key] = {
                "rms_change": (value - baseline[key])
                .flatten(1)
                .square()
                .mean(1)
                .sqrt()
                .tolist()
            }
            if key in target_values:
                target = target_values[key]
                paths[key]["target_mse_increase"] = (
                    (value - target).flatten(1).square().mean(1)
                    - (baseline[key] - target).flatten(1).square().mean(1)
                ).tolist()
        results.append(
            {
                "camera": region.camera,
                "bounds": [region.top, region.left, region.bottom, region.right],
                "outputs": paths,
            }
        )
    return {
        "method": "occlusion_sensitivity_not_attention",
        "replacement": replacement,
        "forward_calls": 2 + len(regions),
        "regions": results,
    }
