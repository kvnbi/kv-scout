from __future__ import annotations

import torch

NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def orthogonalize(
    matrix: torch.Tensor,
    steps: int = 7,
    coefficients: tuple[float, float, float] = NEWTON_SCHULZ_COEFFICIENTS,
    eps: float = 1e-7,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("orthogonalize expects a 2D matrix")
    a, b, c = coefficients
    x = matrix.to(compute_dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def update_scale(shape: torch.Size) -> float:
    rows, cols = shape
    return max(1.0, rows / cols) ** 0.5


class NorMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        newton_schulz_steps: int = 7,
        newton_schulz_dtype: torch.dtype = torch.bfloat16,
        neuron_beta: float = 0.95,
        eps: float = 1e-8,
        cautious: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if not 0.0 <= neuron_beta < 1.0:
            raise ValueError("neuron_beta must lie in [0, 1)")
        if newton_schulz_steps < 1:
            raise ValueError("newton_schulz_steps must be at least 1")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            newton_schulz_steps=newton_schulz_steps,
            newton_schulz_dtype=newton_schulz_dtype,
            neuron_beta=neuron_beta,
            eps=eps,
            cautious=cautious,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError("NorMuon only accepts 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if not state:
                    state["momentum"] = torch.zeros_like(grad)
                    state["neuron"] = torch.zeros(
                        grad.shape[0], device=grad.device, dtype=torch.float32
                    )

                buffer = state["momentum"]
                buffer.mul_(group["momentum"]).add_(grad)
                direction = (
                    grad.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )

                update = orthogonalize(
                    direction,
                    steps=group["newton_schulz_steps"],
                    compute_dtype=group["newton_schulz_dtype"],
                ).float()

                neuron = state["neuron"]
                neuron.mul_(group["neuron_beta"]).add_(
                    update.pow(2).mean(dim=1), alpha=1.0 - group["neuron_beta"]
                )
                normalised = update / (neuron.sqrt().add(group["eps"]).unsqueeze(1))
                scale = update.norm() / normalised.norm().clamp_min(group["eps"])
                update = (normalised * scale).to(param.dtype)

                if group["weight_decay"] > 0.0:
                    decay = param
                    if group["cautious"]:
                        decay = torch.where(
                            update.sign() == param.sign(), param, torch.zeros_like(param)
                        )
                    param.add_(decay, alpha=-group["lr"] * group["weight_decay"])

                param.add_(update, alpha=-group["lr"] * update_scale(param.shape))

        return loss


class CombinedOptimizer:
    def __init__(self, matrix: torch.optim.Optimizer, vector: torch.optim.Optimizer):
        self.matrix = matrix
        self.vector = vector

    @property
    def param_groups(self):
        return self.matrix.param_groups + self.vector.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.matrix.zero_grad(set_to_none=set_to_none)
        self.vector.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.matrix.step()
        self.vector.step()

    def state_dict(self) -> dict:
        return {"matrix": self.matrix.state_dict(), "vector": self.vector.state_dict()}

    def load_state_dict(self, payload: dict) -> None:
        self.matrix.load_state_dict(payload["matrix"])
        self.vector.load_state_dict(payload["vector"])


def group_by_lr_scale(params: list) -> list[dict]:
    groups: dict[float, list] = {}
    for param in params:
        groups.setdefault(float(getattr(param, "mup_lr_scale", 1.0)), []).append(param)
    return [{"params": items, "mup_scale": scale} for scale, items in sorted(groups.items())]


def split_parameters(model) -> tuple[list, list]:
    matrices, vectors = [], []
    excluded = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            excluded.add(f"{name}.weight" if name else "weight")
    head = getattr(model, "head", None)
    if head is not None:
        excluded.add("head.weight")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and name not in excluded:
            matrices.append(param)
        else:
            vectors.append(param)
    return matrices, vectors


def build_optimizer(model, cfg):
    vector_kwargs = dict(
        lr=cfg.peak_lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        cautious=cfg.cautious_weight_decay,
    )
    if cfg.matrix_optimizer == "adamw":
        optimizer = CautiousAdamW(
            group_by_lr_scale(list(model.parameters())), **vector_kwargs
        )
        for group in optimizer.param_groups:
            group["lr_scale"] = group.get("mup_scale", 1.0)
        return optimizer

    if cfg.matrix_optimizer != "normuon":
        raise ValueError(f"unsupported matrix optimizer {cfg.matrix_optimizer}")

    matrices, vectors = split_parameters(model)
    matrix = NorMuon(
        group_by_lr_scale(matrices),
        lr=cfg.peak_lr * cfg.matrix_lr_multiplier,
        momentum=cfg.betas[0],
        weight_decay=cfg.weight_decay,
        newton_schulz_steps=cfg.newton_schulz_steps,
        newton_schulz_dtype=getattr(torch, cfg.newton_schulz_dtype),
        eps=cfg.eps,
        cautious=cfg.cautious_weight_decay,
    )
    vector = CautiousAdamW(group_by_lr_scale(vectors), **vector_kwargs)
    for group in matrix.param_groups:
        group["lr_scale"] = cfg.matrix_lr_multiplier * group.get("mup_scale", 1.0)
    for group in vector.param_groups:
        group["lr_scale"] = group.get("mup_scale", 1.0)
    return CombinedOptimizer(matrix, vector)


class CautiousAdamW(torch.optim.AdamW):
    def __init__(self, params, weight_decay: float = 0.0, cautious: bool = True, **kwargs):
        super().__init__(params, weight_decay=0.0, **kwargs)
        for group in self.param_groups:
            group["decoupled_decay"] = weight_decay
            group["cautious"] = cautious

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group in self.param_groups:
            decay = group["decoupled_decay"]
            if decay <= 0.0:
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                momentum = self.state[param].get("exp_avg")
                if momentum is None:
                    continue
                if group["cautious"]:
                    agrees = (momentum * param) > 0
                    pull = torch.where(agrees, param, torch.zeros_like(param))
                else:
                    pull = param
                param.add_(pull, alpha=-group["lr"] * decay)
        return loss

    def cautious_fraction(self) -> float:
        agreeing = total = 0
        for group in self.param_groups:
            for param in group["params"]:
                momentum = self.state[param].get("exp_avg")
                if momentum is None:
                    continue
                agreeing += int(((momentum * param) > 0).sum())
                total += param.numel()
        return agreeing / total if total else 0.0
