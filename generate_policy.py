from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from remask_policy import (
    BaseBlockGenerator,
    GenerationContext,
    PolicyGuidedGenerator,
    RemaskPolicyMLP,
    RolloutCollector,
    StateTensorEncoder,
    build_reward_adapter,
    build_dataset_bundle,
    load_config,
)
from remask_policy.logging_utils import setup_logger
from remask_policy.rollout import load_prompt_examples, save_rollout_bundles
from remask_policy.trainer import RemaskPolicyTrainer, load_trained_policy
from remask_policy.utils import ensure_dir


def _configure_torch_runtime() -> None:
    try:
        import torch._dynamo
    except Exception:
        return

    torch._dynamo.config.suppress_errors = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone entrypoint for learned remask policy workflows."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["collect", "train", "infer"],
        help="Which remask policy config schema to load.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a remask policy YAML config file.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config as JSON after validation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_torch_runtime()
    args = parse_args()
    logger = setup_logger(level=args.log_level)

    config = load_config(args.config, args.mode)
    logger.info("Loaded %s remask policy config from %s", args.mode, args.config)
    logger.info(
        "Policy mode=%s model_dir=%s output_dir=%s",
        config.policy_mode,
        config.model_dir,
        config.output_dir,
    )

    if args.print_config:
        print(config.to_json())
        return 0

    if args.mode == "collect":
        return _run_collect(config, logger)
    if args.mode == "train":
        return _run_train(config, logger)
    if args.mode == "infer":
        return _run_infer(config, logger)

    return 0


def _run_collect(config, logger) -> int:
    generator = BaseBlockGenerator(config)
    reward_adapter = build_reward_adapter(
        getattr(config, "reward_type", "dummy"),
        pattern=getattr(config, "reward_pattern", None),
    )
    collector = RolloutCollector(
        generator,
        reward_adapter,
        num_counterfactual_blocks=config.num_counterfactual_blocks,
        remask_penalty_lambda=config.remask_penalty_lambda,
        random_seed=config.random_seed,
    )
    examples = load_prompt_examples(config.prompts_path, max_samples=config.max_samples)
    logger.info("Loaded %d prompt examples from %s", len(examples), config.prompts_path)
    bundles = collector.collect_many(examples)
    export_result = save_rollout_bundles(
        bundles,
        config.output_dir,
        config.save_filename,
    )
    logger.info(
        "Saved %d block-level supervision samples to %s",
        export_result.num_supervision_samples,
        export_result.supervision_dataset_path,
    )
    logger.info("Saved %d base rollouts to %s", export_result.num_base_rollouts, export_result.base_rollout_path)
    logger.info(
        "Saved %d single-intervention rollouts to %s",
        export_result.num_branch_rollouts,
        export_result.branch_rollout_path,
    )
    logger.info("Saved feature schema metadata to %s", export_result.feature_schema_path)
    return 0


def _run_train(config, logger) -> int:
    dataset_bundle = build_dataset_bundle(
        config.train_data_path,
        eval_data_path=config.eval_data_path,
        feature_schema_path=config.feature_schema_path,
        feature_names=config.feature_names,
        val_split=config.val_split,
        random_seed=config.random_seed,
        normalize_features=config.normalize_features,
    )
    logger.info(
        "Loaded %d train samples and %d val samples from block supervision data",
        len(dataset_bundle.train_samples),
        len(dataset_bundle.val_samples),
    )

    model = RemaskPolicyMLP(
        input_dim=dataset_bundle.feature_schema.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    trainer = RemaskPolicyTrainer(
        config,
        model,
        dataset_bundle.feature_schema,
        dataset_bundle.normalization_stats,
    )
    summary = trainer.train(
        dataset_bundle.train_dataset,
        dataset_bundle.val_dataset,
    )
    logger.info("Saved policy checkpoint to %s", summary.checkpoint_path)
    logger.info("Saved training metrics to %s", summary.metrics_path)
    logger.info(
        "Best epoch=%d best_score=%.6f feature_dim=%d",
        summary.best_epoch,
        summary.best_score,
        dataset_bundle.feature_schema.input_dim,
    )
    return 0


def _run_infer(config, logger) -> int:
    prompt_text = config.prompt
    if config.prompt_file is not None:
        prompt_text = Path(config.prompt_file).read_text(encoding="utf-8")
    if prompt_text is None:
        raise ValueError("Infer mode requires prompt or prompt_file.")

    context = GenerationContext.from_config(
        config,
        prompt_id="infer_prompt",
        prompt_text=prompt_text,
    )
    base_generator = BaseBlockGenerator(config)
    policy_model = None
    state_tensor_encoder = None
    if config.policy_mode == "learned":
        map_location = config.device
        if map_location.startswith("cuda") and not torch.cuda.is_available():
            map_location = "cpu"
        policy_model, feature_schema, normalization_stats, _ = load_trained_policy(
            config.policy_ckpt,
            map_location=map_location,
        )
        policy_device = config.device
        if policy_device.startswith("cuda") and not torch.cuda.is_available():
            policy_device = "cpu"
        policy_model.to(policy_device)
        state_tensor_encoder = StateTensorEncoder(
            feature_names=feature_schema.feature_names,
            mean=normalization_stats.mean,
            std=normalization_stats.std,
            normalize=normalization_stats.enabled,
            device=policy_device,
        )

    generator = PolicyGuidedGenerator(
        base_generator,
        policy_mode=config.policy_mode,
        policy_model=policy_model,
        state_tensor_encoder=state_tensor_encoder,
        policy_threshold=config.policy_threshold,
        heuristic_confidence_threshold=config.heuristic_confidence_threshold,
    )
    result = generator.generate(context)

    output_dir = ensure_dir(config.output_dir)
    output_path = output_dir / config.save_filename
    output_path.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "generation_result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved policy-guided generation result with %d blocks to %s",
        len(result.blocks),
        output_path,
    )
    logger.info(result.generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
