"""Standalone scaffolding for learned remask policy experiments."""

from .config import (
    BaseRemaskConfig,
    RemaskCollectConfig,
    RemaskInferConfig,
    RemaskTrainConfig,
    load_config,
    save_config,
)
from .block_runner import BaseBlockGenerator, BlockGenerationResult, GenerationContext
from .dataset import (
    BlockSupervisionDataset,
    FeatureSchema,
    NormalizationStats,
    RemaskDatasetBundle,
    build_dataset_bundle,
    load_feature_schema,
    load_supervision_samples,
)
from .interfaces import (
    BlockDecisionRecord,
    PolicyDecision,
    RewardResult,
    RolloutRecord,
)
from .inference import PolicyGuidedGenerationResult, PolicyGuidedGenerator, RuntimeBlockRecord
from .reward import (
    DummyRewardAdapter,
    ExactMatchRewardAdapter,
    FormatValidityRewardAdapter,
    HookRewardAdapter,
    build_reward_adapter,
)
from .policy_net import RemaskPolicyMLP
from .rollout import (
    PromptExample,
    RolloutBundle,
    RolloutCollector,
    RolloutExportResult,
)
from .state_encoder import (
    DEFAULT_BLOCK_FEATURE_DESCRIPTIONS,
    DEFAULT_BLOCK_FEATURE_NAMES,
    StateTensorEncoder,
    build_block_state_features,
)
from .trainer import RemaskPolicyTrainer, load_trained_policy

__all__ = [
    "BaseRemaskConfig",
    "RemaskCollectConfig",
    "RemaskInferConfig",
    "RemaskTrainConfig",
    "BaseBlockGenerator",
    "GenerationContext",
    "BlockGenerationResult",
    "FeatureSchema",
    "NormalizationStats",
    "BlockSupervisionDataset",
    "RemaskDatasetBundle",
    "build_dataset_bundle",
    "load_feature_schema",
    "load_supervision_samples",
    "BlockDecisionRecord",
    "PolicyDecision",
    "RewardResult",
    "RolloutRecord",
    "PolicyGuidedGenerator",
    "PolicyGuidedGenerationResult",
    "RuntimeBlockRecord",
    "DummyRewardAdapter",
    "ExactMatchRewardAdapter",
    "FormatValidityRewardAdapter",
    "HookRewardAdapter",
    "build_reward_adapter",
    "RemaskPolicyMLP",
    "DEFAULT_BLOCK_FEATURE_NAMES",
    "DEFAULT_BLOCK_FEATURE_DESCRIPTIONS",
    "StateTensorEncoder",
    "build_block_state_features",
    "PromptExample",
    "RolloutBundle",
    "RolloutCollector",
    "RolloutExportResult",
    "RemaskPolicyTrainer",
    "load_trained_policy",
    "load_config",
    "save_config",
]
