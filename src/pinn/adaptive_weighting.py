"""Adaptive loss weighting for PINN training.

The default static-weight scheme (``lambda_data=10``, ``lambda_pde=1``,
``lambda_bc=lambda_ic=5`` from ``configs/base.yaml``) is the textbook
PINN failure mode: the data term — which has gradients orders of
magnitude larger than the PDE residual — dominates and the network
converges to a function that fits the data but violates the physics.
Phase A1 of the publication roadmap (``docs/plan_publication_roadmap.md``)
calls this out explicitly.

This module provides two strategies that the trainer can plug in
without changing its overall structure:

* :class:`SoftAdapt` (Heydari et al., 2019, "SoftAdapt: Techniques for
  Adaptive Loss Weighting of Neural Networks with Multi-Part Loss
  Functions"). Cheap, no extra autograd cost, surprisingly effective
  for PINNs (used by e.g. modulus-sym).
* :class:`GradNorm` (Chen et al., 2018, "GradNorm: Gradient Normalization
  for Adaptive Loss Balancing"). More principled but each step requires
  ``len(loss_terms)`` extra backward passes. Use only when SoftAdapt
  doesn't converge.

NTK-based weighting (Wang et al., 2022) is the gold standard but
requires assembling a per-iteration NTK diagonal which is too expensive
on the LMB problem at our scale; we leave it out.

Both strategies are stateful — instantiate once, call :meth:`update` at
the end of every iteration with the current per-term losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SoftAdapt
# ---------------------------------------------------------------------------


@dataclass
class SoftAdaptConfig:
    """Knobs for :class:`SoftAdapt`.

    Args:
        beta: temperature in the softmax over loss-relative-changes.
            Larger ``beta`` makes weights chase the noisiest term more
            aggressively. The original paper uses 0.1.
        normalised: if True, divides each per-term loss by its own
            running mean before computing the relative change. Stops
            terms with very different magnitudes from drowning each
            other out (recommended for PINNs).
        warmup_steps: number of iterations during which weights are
            held at uniform 1/N. Lets transients settle.
        ema_alpha: exponential moving-average factor for the smoothed
            per-term losses (the "previous" loss in the formula).
            ``0.0`` recovers the original paper's instantaneous form.
        eps: numerical floor.
    """

    beta: float = 0.1
    normalised: bool = True
    warmup_steps: int = 50
    ema_alpha: float = 0.9
    eps: float = 1e-8


class SoftAdapt:
    """Heydari-style adaptive weighting.

    Usage::

        adapter = SoftAdapt(["data", "pde", "bc", "ic"])
        for step in range(...):
            losses = compute_losses(...)
            weights = adapter.weights()
            total = sum(weights[k] * losses[k] for k in losses)
            total.backward(); optimizer.step()
            adapter.update({k: float(v.detach()) for k, v in losses.items()})

    The adapter never holds tensors — only Python floats — so it is
    immune to leaking the autograd graph.
    """

    def __init__(
        self,
        loss_names: list[str],
        config: Optional[SoftAdaptConfig] = None,
    ):
        if not loss_names:
            raise ValueError("loss_names must be non-empty")
        self.loss_names = list(loss_names)
        self.cfg = config or SoftAdaptConfig()
        self._step = 0
        self._prev: dict[str, float] = {n: 1.0 for n in self.loss_names}
        self._weights: dict[str, float] = {
            n: 1.0 / len(self.loss_names) for n in self.loss_names
        }

    def weights(self) -> dict[str, float]:
        """Return the current weights dict (one entry per loss name)."""
        return dict(self._weights)

    @property
    def step_count(self) -> int:
        return self._step

    def update(self, losses: dict[str, float]) -> None:
        """Recompute weights given the latest per-term losses.

        Missing names are skipped; extra names raise.
        """
        for name in losses:
            if name not in self._weights:
                raise KeyError(
                    f"SoftAdapt was constructed without loss term '{name}'"
                )
        self._step += 1

        if self._step <= self.cfg.warmup_steps:
            # Hold uniform weights during warmup.
            self._update_ema(losses)
            return

        deltas: dict[str, float] = {}
        for name, loss in losses.items():
            prev = max(self._prev.get(name, 1.0), self.cfg.eps)
            curr = max(loss, self.cfg.eps)
            if self.cfg.normalised:
                deltas[name] = (curr - prev) / prev
            else:
                deltas[name] = curr - prev

        # Numerically-stable softmax over deltas.
        d_max = max(deltas.values())
        exps = {k: math.exp(self.cfg.beta * (v - d_max)) for k, v in deltas.items()}
        Z = sum(exps.values()) + self.cfg.eps
        self._weights = {k: e / Z for k, e in exps.items()}

        self._update_ema(losses)

    def _update_ema(self, losses: dict[str, float]) -> None:
        a = self.cfg.ema_alpha
        for name, loss in losses.items():
            self._prev[name] = a * self._prev[name] + (1.0 - a) * float(loss)


# ---------------------------------------------------------------------------
# GradNorm (optional, more expensive)
# ---------------------------------------------------------------------------


class GradNorm(nn.Module):
    """Chen-2018 gradient-norm balancing.

    Each loss term gets a learnable scalar weight. After the main
    backward pass we run one extra backward per loss to obtain
    ``||∇w_i L_i||`` and update weights so that gradients of all terms
    stay close to a target. Use only when SoftAdapt fails to converge.

    Memory cost: O(N) extra backward passes per step.

    Reference: Chen et al., ICML 2018, "GradNorm".
    """

    def __init__(
        self,
        loss_names: list[str],
        alpha: float = 1.5,
        init_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_names = list(loss_names)
        self.alpha = alpha
        self.weights = nn.Parameter(
            torch.full((len(loss_names),), float(init_weight))
        )
        self._initial_losses: Optional[torch.Tensor] = None

    def weights_dict(self) -> dict[str, float]:
        with torch.no_grad():
            return {n: float(w) for n, w in zip(self.loss_names, self.weights)}

    def gradnorm_loss(
        self,
        per_term_losses: dict[str, torch.Tensor],
        shared_parameters: list[nn.Parameter],
    ) -> torch.Tensor:
        """Compute the auxiliary loss whose gradient updates the weights.

        Caller is responsible for stepping a separate optimiser on
        ``self.weights`` with this loss; the main task loss must be
        computed and backwarded *before* calling this.
        """
        loss_vec = torch.stack(
            [per_term_losses[n] for n in self.loss_names]
        ).detach()
        if self._initial_losses is None:
            self._initial_losses = loss_vec.clone()
        ratios = loss_vec / (self._initial_losses + 1e-12)
        avg_ratio = ratios.mean()
        targets = (ratios / (avg_ratio + 1e-12)) ** self.alpha

        norms = []
        for n in self.loss_names:
            g = torch.autograd.grad(
                self.weights[self.loss_names.index(n)] * per_term_losses[n].detach(),
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            flat = torch.cat(
                [gi.flatten() for gi in g if gi is not None]
            ) if any(gi is not None for gi in g) else torch.zeros(1, device=loss_vec.device)
            norms.append(flat.norm())
        norm_vec = torch.stack(norms)
        norm_avg = norm_vec.mean().detach()
        return (norm_vec - norm_avg * targets).abs().sum()


# ---------------------------------------------------------------------------
# Output-side hard boundary condition wrap
# ---------------------------------------------------------------------------


class HardBoundaryVoltage(nn.Module):
    """Wrap an MLP head so the predicted voltage trajectory hits the
    cell's start- and end-voltage targets exactly, regardless of network
    output. Keeps the soft data-loss honest by removing the trivial
    "endpoint match" failure mode where the network learns the average
    voltage and misses the curvature.

    For a normalised time ``t ∈ [0, 1]`` and base predictions
    ``v_init(params)`` and ``v_final(params)`` (e.g. cutoff voltages
    from the cell spec), the wrapped output is::

        V(t) = (1 - t) * v_init + t * v_final + t * (1 - t) * NN(t, params)

    The bump factor ``t * (1 - t)`` vanishes at both endpoints so the
    boundary conditions are enforced architecturally rather than via a
    soft loss.
    """

    def __init__(self, base_module: nn.Module):
        super().__init__()
        self.base = base_module

    def forward(
        self,
        t: torch.Tensor,
        params: torch.Tensor,
        v_init: torch.Tensor,
        v_final: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if v_init.dim() == 1:
            v_init = v_init.unsqueeze(-1)
        if v_final.dim() == 1:
            v_final = v_final.unsqueeze(-1)
        nn_out = self.base(t, params)
        if nn_out.dim() == 1:
            nn_out = nn_out.unsqueeze(-1)
        bump = t * (1.0 - t)
        return (1.0 - t) * v_init + t * v_final + bump * nn_out
