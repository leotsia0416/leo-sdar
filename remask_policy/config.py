from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from .utils import SerializableMixin, dump_yaml_file, load_yaml_file

VALID_DTYPES = frozenset({"float16", "bfloat16", "float32"})
CONFIG_MODES = frozenset({"collect", "train", "infer"})
REMASKING_STRATEGIES = frozenset(
    {"low_confidence_dynamic", "low_confidence_static", "sequential", "entropy_bounded"}
)

ConfigT = TypeVar("ConfigT", bound="BaseRemaskConfig")


@dataclass
class BaseRemaskConfig(SerializableMixin):
    model_dir: str
    trust_remote_code: bool = False
    device: str = "cuda"
    dtype: str = "float16"
    prompt_length: int = 4096
    block_length: int = 4
    gen_length: int = 128
    denoising_steps: int = 4
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    remasking_strategy: str = "low_confidence_dynamic"
    confidence_threshold: float = 0.85
    eb_threshold: float | None = 0.35
    stopping_criteria_idx: list[int] | None = None
    num_counterfactual_blocks: int = 1
    remask_penalty_lambda: float = 0.0
    policy_mode: str = "disabled"
    policy_ckpt: str | None = None
    policy_threshold: float = 0.5
    output_dir: str = "outputs/remask_policy"
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.model_dir:
            raise ValueError("model_dir must be set.")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(VALID_DTYPES)}.")
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive.")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive.")
        if self.gen_length <= 0:
            raise ValueError("gen_length must be positive.")
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if self.remasking_strategy not in REMASKING_STRATEGIES:
            raise ValueError(
                f"remasking_strategy must be one of {sorted(REMASKING_STRATEGIES)}."
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if self.eb_threshold is not None and self.eb_threshold < 0:
            raise ValueError("eb_threshold must be non-negative when provided.")
        if self.num_counterfactual_blocks <= 0:
            raise ValueError("num_counterfactual_blocks must be positive.")
        if not 0.0 <= self.policy_threshold <= 1.0:
            raise ValueError("policy_threshold must be between 0 and 1.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if not self.output_dir:
            raise ValueError("output_dir must be set.")

    @classmethod
    def load_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        return cls.from_dict(load_yaml_file(path))

    def save_yaml(self, path: str | Path) -> None:
        dump_yaml_file(path, self.to_dict())


@dataclass
class RemaskCollectConfig(BaseRemaskConfig):
    prompts_path: str = "data/remask/prompts.jsonl"
    max_samples: int | None = None
    save_filename: str = "remask_collect.jsonl"
    reward_type: str = "dummy"
    reward_pattern: str | None = None


@dataclass
class RemaskTrainConfig(BaseRemaskConfig):
    train_data_path: str = "outputs/remask_policy/collect/remask_collect.jsonl"
    eval_data_path: str | None = None
    feature_schema_path: str | None = None
    feature_names: list[str] = field(default_factory=list)
    val_split: float = 0.1
    normalize_features: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 3
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    num_workers: int = 0
    checkpoint_name: str = "policy.pt"
    metrics_filename: str = "train_metrics.json"

    def validate(self) -> None:
        super().validate()
        if self.val_split < 0.0 or self.val_split >= 1.0:
            raise ValueError("val_split must be in [0, 1).")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("hidden_dims must all be positive.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if not self.checkpoint_name:
            raise ValueError("checkpoint_name must be set.")
        if not self.metrics_filename:
            raise ValueError("metrics_filename must be set.")


@dataclass
class RemaskInferConfig(BaseRemaskConfig):
    heuristic_confidence_threshold: float = 0.5
    prompt: str | None = None
    prompt_file: str | None = None
    save_filename: str = "remask_infer.json"

    def validate(self) -> None:
        super().validate()
        if self.policy_mode not in {"off", "heuristic", "learned"}:
            raise ValueError("policy_mode for inference must be one of: off, heuristic, learned.")
        if not 0.0 <= self.heuristic_confidence_threshold <= 1.0:
            raise ValueError("heuristic_confidence_threshold must be between 0 and 1.")
        if self.policy_mode == "learned" and not self.policy_ckpt:
            raise ValueError("policy_ckpt must be set when policy_mode=learned.")
        if self.prompt and self.prompt_file:
            raise ValueError("Specify either prompt or prompt_file, not both.")


CONFIG_CLASS_BY_MODE = {
    "collect": RemaskCollectConfig,
    "train": RemaskTrainConfig,
    "infer": RemaskInferConfig,
}


def load_config(path: str | Path, mode: str) -> BaseRemaskConfig:
    if mode not in CONFIG_CLASS_BY_MODE:
        raise ValueError(f"mode must be one of {sorted(CONFIG_MODES)}.")
    return CONFIG_CLASS_BY_MODE[mode].load_yaml(path)


def save_config(config: BaseRemaskConfig, path: str | Path) -> None:
    dump_yaml_file(path, config.to_dict())
