from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, get_type_hints


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 32768
    model: str = "byte_level_bpe"
    byte_fallback: bool = True
    reflection_slots: int = 32
    seed_forms: tuple[str, ...] = (
        "colour",
        "realise",
        "centre",
        "travelled",
        "defence",
    )
    spelling: str = "ise"

    def __post_init__(self) -> None:
        if self.vocab_size > 65536:
            raise ValueError("vocab_size must fit in uint16 token shards")
        if self.spelling not in ("ise", "ize"):
            raise ValueError("spelling must be one of ise, ize")
        if self.reflection_slots >= self.vocab_size:
            raise ValueError("reflection_slots must be smaller than vocab_size")

    @property
    def reflection_token_ids(self) -> range:
        return range(self.vocab_size - self.reflection_slots, self.vocab_size)


@dataclass(frozen=True)
class LanguageMix:
    english: float = 0.95
    german: float = 0.05

    def __post_init__(self) -> None:
        total = self.english + self.german
        if abs(total - 1.0) > 1e-6:
            raise ValueError("language weights must sum to 1.0")


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 12
    top_k: int = 2
    shared_experts: int = 1
    expert_ffn_hidden: int = 1280
    shared_ffn_hidden: int = 2560
    routing: str = "sigmoid"
    aux_loss_free_balancing: bool = True
    first_moe_layer: int = 3

    def __post_init__(self) -> None:
        if self.top_k >= self.num_experts:
            raise ValueError("top_k must be smaller than num_experts")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 1920
    n_layers: int = 30
    dense_warmup_layers: int = 2
    attention_anchor_layers: tuple[int, ...] = (6, 9, 12, 15, 18, 21, 27, 30)
    n_query_heads: int = 15
    n_kv_heads: int = 5
    head_dim: int = 128
    context_min: int = 1024
    context_max: int = 8192
    tie_embeddings: bool = True
    mtp_heads: int = 2
    rope_on_linear_layers: bool = True
    nope_on_anchor_layers: bool = True
    cross_layer_kv_sharing: bool = True
    attention_sinks: bool = True
    qk_norm: bool = True
    normalized_value_residual: bool = True
    layernorm_scaling: bool = True
    per_head_gated_attention: bool = True
    use_gdn: bool = True
    use_moe: bool = True
    dense_ffn_hidden: int = 5120
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    precision: str = "bfloat16"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    languages: LanguageMix = field(default_factory=LanguageMix)

    def __post_init__(self) -> None:
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")
        if self.n_query_heads * self.head_dim != self.d_model:
            raise ValueError("n_query_heads times head_dim must equal d_model")
        if self.dense_warmup_layers >= self.n_layers:
            raise ValueError("dense_warmup_layers must be smaller than n_layers")
        if self.context_min > self.context_max:
            raise ValueError("context_min must not exceed context_max")
        for layer in self.attention_anchor_layers:
            if not 1 <= layer <= self.n_layers:
                raise ValueError("anchor layer index out of range")
            if layer <= self.dense_warmup_layers:
                raise ValueError("anchor layers must follow the dense warmup layers")
        if len(set(self.attention_anchor_layers)) != len(self.attention_anchor_layers):
            raise ValueError("anchor layer indices must be unique")

    @property
    def n_anchor_layers(self) -> int:
        return len(self.attention_anchor_layers)

    @property
    def n_linear_layers(self) -> int:
        return self.n_layers - self.dense_warmup_layers - self.n_anchor_layers

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def kv_group_size(self) -> int:
        return self.n_query_heads // self.n_kv_heads

    @property
    def linear_to_anchor_ratio(self) -> float:
        if self.n_anchor_layers == 0:
            return float("inf")
        return self.n_linear_layers / self.n_anchor_layers

    def layer_kind(self, layer: int) -> str:
        if not 1 <= layer <= self.n_layers:
            raise ValueError("layer index out of range")
        if layer <= self.dense_warmup_layers:
            return "dense"
        if layer in self.attention_anchor_layers:
            return "anchor"
        return "linear" if self.use_gdn else "attention"

    def ffn_hidden(self, layer: int) -> int:
        if self.layer_kind(layer) == "dense" or not self.use_moe:
            return self.dense_ffn_hidden
        return self.moe.expert_ffn_hidden

    def parameter_estimate(self) -> int:
        d = self.d_model
        embed = self.vocab_size * d
        if not self.tie_embeddings:
            embed *= 2
        total = embed + d
        for layer in range(1, self.n_layers + 1):
            kv = self.n_kv_heads * self.head_dim
            attention = d * d + 2 * d * kv + d * d
            if self.qk_norm:
                attention += 2 * self.head_dim
            if self.normalized_value_residual and layer > 1:
                attention += 1
            if self.per_head_gated_attention:
                attention += d * self.n_query_heads + self.n_query_heads
            if self.layer_kind(layer) == "dense" or not self.use_moe:
                ffn = 3 * d * self.ffn_hidden(layer)
            else:
                experts = self.moe.num_experts * 3 * d * self.moe.expert_ffn_hidden
                shared = self.moe.shared_experts * 3 * d * self.moe.shared_ffn_hidden
                ffn = experts + shared + d * self.moe.num_experts
            total += attention + ffn + 2 * d
        return total

    def active_parameter_estimate(self) -> int:
        if not self.use_moe:
            return self.parameter_estimate()
        d = self.d_model
        idle = 0
        for layer in range(1, self.n_layers + 1):
            if self.layer_kind(layer) == "dense":
                continue
            skipped = self.moe.num_experts - self.moe.top_k
            idle += skipped * 3 * d * self.moe.expert_ffn_hidden
        return self.parameter_estimate() - idle


PROXY_ANCHORS = (6, 9, 12)


def proxy_config(**overrides) -> ModelConfig:
    base = dict(
        d_model=576,
        n_layers=12,
        dense_warmup_layers=2,
        attention_anchor_layers=PROXY_ANCHORS,
        n_query_heads=6,
        n_kv_heads=2,
        head_dim=96,
        context_min=1024,
        context_max=1024,
        dense_ffn_hidden=1536,
        use_gdn=False,
        use_moe=False,
        nope_on_anchor_layers=False,
        cross_layer_kv_sharing=False,
        attention_sinks=False,
        qk_norm=False,
        normalized_value_residual=False,
        layernorm_scaling=False,
        per_head_gated_attention=False,
        mtp_heads=0,
        precision="float32",
        moe=MoEConfig(expert_ffn_hidden=384, shared_ffn_hidden=768),
    )
    base.update(overrides)
    return ModelConfig(**base)


@dataclass(frozen=True)
class OptimConfig:
    matrix_optimizer: str = "normuon"
    vector_optimizer: str = "adamw"
    newton_schulz_steps: int = 7
    newton_schulz_dtype: str = "bfloat16"
    matrix_lr_multiplier: float = 30.0
    peak_lr: float = 3e-3
    min_lr_fraction: float = 0.0
    weight_decay: float = 0.1
    cautious_weight_decay: bool = True
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    z_loss_weight: float = 1e-4
    schedule: str = "wsd"
    warmup_steps: int = 2000
    stable_fraction_of_peak: float = 0.55
    decay_fraction: float = 0.20
    decay_profile: str = "inv_sqrt"
    ema_decay: float = 0.999

    def __post_init__(self) -> None:
        if not 0.0 < self.decay_fraction < 1.0:
            raise ValueError("decay_fraction must lie strictly between 0 and 1")
        if not 0.0 < self.stable_fraction_of_peak <= 1.0:
            raise ValueError("stable_fraction_of_peak must lie in (0, 1]")
        if self.schedule not in ("wsd", "constant"):
            raise ValueError("schedule must be one of wsd, constant")


@dataclass(frozen=True)
class DataConfig:
    shard_dir: str = "data/shards"
    index_name: str = "index.json"
    seq_len: int = 1024
    batch_size: int = 8
    shuffle: bool = True
    seed: int = 7
    drop_last: bool = True

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    checkpoint_every: int = 50
    log_every: int = 1
    keep_last_checkpoints: int = 3
    seed: int = 1234
    device: str = "auto"
    dtype: str = "float32"
    out_dir: str = "runs/default"
    kill_at: int | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if self.keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be at least 1")


@dataclass(frozen=True)
class HarnessModelConfig:
    vocab_size: int = 2048
    d_model: int = 288
    n_layers: int = 5
    n_heads: int = 9
    ffn_hidden: int = 672
    seq_len: int = 256
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab_size > 65536:
            raise ValueError("vocab_size must fit in uint16 token shards")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def param_count(self) -> int:
        embed = self.vocab_size * self.d_model
        if not self.tie_embeddings:
            embed *= 2
        per_layer = (
            3 * self.d_model * self.d_model
            + self.d_model * self.d_model
            + 3 * self.d_model * self.ffn_hidden
            + 2 * self.d_model
        )
        return embed + self.n_layers * per_layer + self.d_model


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass(frozen=True)
class HarnessRunConfig:
    model: HarnessModelConfig = field(default_factory=HarnessModelConfig)
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            matrix_optimizer="adamw",
            peak_lr=3e-3,
            warmup_steps=20,
            z_loss_weight=0.0,
        )
    )
    data: DataConfig = field(
        default_factory=lambda: DataConfig(seq_len=256, batch_size=8)
    )
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            steps=200, checkpoint_every=10, out_dir="runs/harness"
        )
    )


def to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    raise TypeError("to_dict expects a dataclass instance")


def from_dict(cls: type, payload: dict) -> Any:
    if not is_dataclass(cls):
        raise TypeError("from_dict expects a dataclass type")
    resolved = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        if name not in cls.__dataclass_fields__:
            continue
        target = resolved.get(name)
        if isinstance(value, dict) and is_dataclass(target):
            kwargs[name] = from_dict(target, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
