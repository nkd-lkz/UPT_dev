# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Causally summarize command-response dynamics as a compact physical token."""

from __future__ import annotations

import torch


class CausalResponseMatrix:
    """Estimate a local 3-D execution map from completed transitions.

    Row-vector displacements follow ``measured = expected @ matrix``. A ridge
    prior starts the map at identity, so no correction is implied before the
    robot has observed a response. ``observe`` must be called only after the
    corresponding command has finished executing.
    """

    def __init__(self, ridge: float = 1e-3) -> None:
        if not torch.isfinite(torch.tensor(ridge)) or ridge <= 0:
            raise ValueError("Response-matrix ridge must be finite and positive")
        self.ridge = float(ridge)
        self._gram = torch.eye(3, dtype=torch.float64) * ridge
        self._cross = torch.eye(3, dtype=torch.float64) * ridge
        self.observations = 0

    def observe(self, expected: torch.Tensor, measured: torch.Tensor) -> None:
        """Update the map from one completed expected/measured displacement."""
        expected = torch.as_tensor(expected, dtype=torch.float64).flatten()
        measured = torch.as_tensor(measured, dtype=torch.float64).flatten()
        if (
            expected.shape != (3,)
            or measured.shape != (3,)
            or not torch.isfinite(expected).all()
            or not torch.isfinite(measured).all()
        ):
            raise ValueError("Response observations must be finite 3-D vectors")
        self._gram += torch.outer(expected, expected)
        self._cross += torch.outer(expected, measured)
        self.observations += 1

    def matrix(self) -> torch.Tensor:
        """Return the current 3×3 expected-to-measured response map."""
        return torch.linalg.solve(self._gram, self._cross).float()

    def support(self) -> torch.Tensor:
        """Return dimensionless excitation evidence for each input direction."""
        evidence = (self._gram.diag() - self.ridge).clamp_min(0)
        return torch.log1p(evidence / self.ridge).float()

    def token(self) -> torch.Tensor:
        """Return a 12-D physical token: response map plus axis support."""
        token = torch.cat((self.matrix().flatten(), self.support()))
        if token.shape != (12,) or not torch.isfinite(token).all():
            raise FloatingPointError("Invalid causal response token")
        return token

    def predict(self, expected: torch.Tensor) -> torch.Tensor:
        """Predict measured displacement under the current causal estimate."""
        expected = torch.as_tensor(expected, dtype=torch.float32).flatten()
        if expected.shape != (3,) or not torch.isfinite(expected).all():
            raise ValueError("Expected displacement must be a finite 3-D vector")
        return expected @ self.matrix()

    def correction(
        self,
        *,
        minimum_observations: int = 2,
        singular_bounds: tuple[float, float] = (0.8, 1.3),
        reversed_control: bool = False,
    ) -> torch.Tensor:
        """Return a bounded command map using only completed observations.

        The causal arm applies the inverse expected-to-measured map. The
        deliberately wrong control applies the map itself. Singular-value
        bounds preserve the reference action prior while allowing cross-axis
        corrections.
        """
        low, high = singular_bounds
        if (
            minimum_observations < 1
            or not 0 < low <= 1 <= high
            or not torch.isfinite(torch.tensor(singular_bounds)).all()
        ):
            raise ValueError("Invalid response-matrix correction contract")
        if self.observations < minimum_observations:
            return torch.eye(3)
        response = self.matrix()
        raw = response if reversed_control else torch.linalg.inv(response)
        left, singular, right = torch.linalg.svd(raw)
        bounded = (left * singular.clamp(low, high)) @ right
        if not torch.isfinite(bounded).all():
            raise FloatingPointError("Invalid bounded response correction")
        return bounded
