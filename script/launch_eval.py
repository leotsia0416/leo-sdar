#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "script" / "config" / "experiments"
RUNNER_PATHS = {
    "gap": REPO_ROOT / "script" / "test_gap.sh",
    "lmdeploy": REPO_ROOT / "script" / "test.sh",
    "base_hf": REPO_ROOT / "script" / "experiments" / "test_base.sh",
}
SUBMIT_COMMANDS = {
    "sbatch": "sbatch",
    "bash": "bash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SDAR eval jobs from experiment YAML manifests."
    )
    parser.add_argument(
        "experiment",
        help="Experiment yaml path, or a file name under script/config/experiments/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved jobs without dispatching them.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only launch jobs with these names.",
    )
    return parser.parse_args()


def resolve_experiment_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    fallback = (EXPERIMENTS_DIR / candidate.name).resolve()
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"Cannot find experiment manifest: {raw} (also tried {fallback})"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment manifest must be a mapping: {path}")
    return payload


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_value(item, context) for key, item in value.items()}
    return value


def normalize_env(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    formatted = format_value(payload, context)
    env: dict[str, str] = {}
    for key, value in formatted.items():
        if value is None:
            continue
        if isinstance(value, bool):
            env[key] = "true" if value else "false"
        else:
            env[key] = str(value)
    return env


def build_jobs(manifest: dict[str, Any], manifest_path: Path, only: set[str] | None) -> list[dict[str, Any]]:
    experiment_name = manifest.get("name") or manifest_path.stem
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_context = {
        "repo_root": str(REPO_ROOT),
        "experiment_name": experiment_name,
        "run_stamp": run_stamp,
    }
    resolved_global_vars = format_value(manifest.get("vars") or {}, base_context)
    global_context = {
        **base_context,
        **resolved_global_vars,
    }

    base_job = {
        "runner": manifest.get("runner", "gap"),
        "submit_mode": manifest.get("submit_mode", "sbatch"),
        "env": manifest.get("shared_env", {}),
        "unset_env": manifest.get("unset_env", []),
    }
    raw_jobs = manifest.get("jobs") or []
    if not raw_jobs:
        raw_jobs = [{"name": experiment_name}]

    jobs: list[dict[str, Any]] = []
    for idx, raw_job in enumerate(raw_jobs, start=1):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Job entry #{idx} must be a mapping in {manifest_path}")
        job = merge_dict(base_job, raw_job)
        job_name = job.get("name") or f"{experiment_name}_{idx:02d}"
        if only and job_name not in only:
            continue

        resolved_job_vars = format_value(job.get("vars") or {}, global_context)
        context = {
            **global_context,
            "job_index": idx,
            "job_name": job_name,
            **resolved_job_vars,
        }
        runner_key = format_value(job.get("runner", "gap"), context)
        submit_mode = format_value(job.get("submit_mode", "sbatch"), context)
        env = normalize_env(job.get("env", {}), context)
        unset_env = [str(item) for item in format_value(job.get("unset_env", []), context)]

        jobs.append(
            {
                "name": job_name,
                "runner": runner_key,
                "submit_mode": submit_mode,
                "env": env,
                "unset_env": unset_env,
                "context": context,
            }
        )

    return jobs


def shell_preview(job: dict[str, Any], runner_path: Path) -> str:
    env_parts = []
    for key, value in sorted(job["env"].items()):
        env_parts.append(f"{key}={shlex.quote(value)}")
    for key in job["unset_env"]:
        env_parts.append(f"unset {key};")
    command = [SUBMIT_COMMANDS[job["submit_mode"]], str(runner_path)]
    return " ".join(env_parts + [shlex.join(command)])


def dispatch_job(job: dict[str, Any], runner_path: Path) -> None:
    env = os.environ.copy()
    env.update(job["env"])
    for key in job["unset_env"]:
        env.pop(key, None)

    command = [SUBMIT_COMMANDS[job["submit_mode"]], str(runner_path)]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)


def main() -> int:
    args = parse_args()
    manifest_path = resolve_experiment_path(args.experiment)
    manifest = load_yaml(manifest_path)
    only = set(args.only) if args.only else None
    jobs = build_jobs(manifest, manifest_path, only)

    if not jobs:
        raise SystemExit("No jobs matched the experiment manifest.")

    experiment_name = manifest.get("name") or manifest_path.stem
    print(f"Experiment: {experiment_name}")
    print(f"Manifest: {manifest_path}")
    print(f"Jobs: {len(jobs)}")

    for idx, job in enumerate(jobs, start=1):
        runner_key = job["runner"]
        if runner_key not in RUNNER_PATHS:
            raise KeyError(
                f"Unsupported runner '{runner_key}'. Choices: {sorted(RUNNER_PATHS)}"
            )
        if job["submit_mode"] not in SUBMIT_COMMANDS:
            raise KeyError(
                f"Unsupported submit_mode '{job['submit_mode']}'. Choices: {sorted(SUBMIT_COMMANDS)}"
            )
        runner_path = RUNNER_PATHS[runner_key]
        preview = shell_preview(job, runner_path)
        print(f"[{idx}/{len(jobs)}] {job['name']}")
        print(f"  runner: {runner_key} -> {runner_path}")
        print(f"  submit: {job['submit_mode']}")
        print(f"  command: {preview}")
        if args.dry_run:
            continue
        dispatch_job(job, runner_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
