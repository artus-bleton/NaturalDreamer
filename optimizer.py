import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class LaPropAGC(Optimizer):
    """
    Implémentation PyTorch de LaProp + Adaptive Gradient Clipping (AGC),
    inspirée de DreamerV3.

    - RMS (beta2) sur les gradients au carré
    - Momentum (beta1) sur le gradient normalisé RMS
    - AGC: clip du gradient en fonction de la norme du paramètre
    - Weight decay optionnel (style AdamW: decoupled)
    """

    def __init__(
        self,
        params,
        lr: float = 4e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        agc_clip: float = 0.3,
        agc_eps: float = 1e-3,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if agc_clip < 0.0:
            raise ValueError(f"Invalid agc_clip: {agc_clip}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            agc_clip=agc_clip,
            agc_eps=agc_eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _apply_agc(self, param, grad, clip, eps):
        """
        AGC: adapative gradient clipping
        update *= max_norm / grad_norm, avec max_norm = clip * max(pnorm, eps)
        """
        if grad is None:
            return

        # ne clip pas les scalaires (ex: bias)
        if grad.ndim <= 1:
            return

        p = param.data
        g = grad.data

        p_norm = p.norm(2)
        g_norm = g.norm(2)

        if p_norm == 0 or g_norm == 0:
            return

        max_norm = clip * max(p_norm, eps)
        if g_norm > max_norm:
            scale = max_norm / (g_norm + 1e-12)
            g.mul_(scale)

    @torch.no_grad()
    def step(self, closure=None) -> float:
        """
        Un step:
        - calcule AGC sur chaque gradient (optionnel)
        - applique LaProp (RMS + momentum + bias correction)
        - applique decoupled weight decay
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            agc_clip = group["agc_clip"]
            agc_eps = group["agc_eps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # AGC (comme clip_by_agc)
                if agc_clip > 0.0:
                    self._apply_agc(p, grad, agc_clip, agc_eps)

                state = self.state[p]

                # State init
                if len(state) == 0:
                    state["step"] = 0
                    state["nu"] = torch.zeros_like(p, memory_format=torch.preserve_format)  # EMA des g^2
                    state["mu"] = torch.zeros_like(p, memory_format=torch.preserve_format)  # EMA des updates

                nu = state["nu"]
                mu = state["mu"]
                step = state["step"] + 1
                state["step"] = step

                # --- scale_by_rms: EMA des carrés de gradients ---
                nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # bias correction
                bias_correction2 = 1.0 - beta2 ** step
                nu_hat = nu / bias_correction2

                # g / (sqrt(nu_hat) + eps)
                scaled_grad = grad / (nu_hat.sqrt() + eps)

                # --- scale_by_momentum: EMA des gradients normalisés RMS ---
                mu.mul_(beta1).add_(scaled_grad, alpha=1.0 - beta1)

                if nesterov:
                    # comme optax: 2x update_moment pour Nesterov
                    mu_n = beta1 * mu + (1.0 - beta1) * scaled_grad
                    bias_correction1 = 1.0 - beta1 ** step
                    update = mu_n / bias_correction1
                else:
                    bias_correction1 = 1.0 - beta1 ** step
                    update = mu / bias_correction1

                # --- decoupled weight decay (style AdamW / optax.add_decayed_weights) ---
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # --- apply update (scale_by_learning_rate) ---
                p.data.add_(update, alpha=-lr)

        return loss
