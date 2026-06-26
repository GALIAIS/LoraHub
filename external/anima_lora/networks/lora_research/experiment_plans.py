"""Research-only LoRA experiment plans.

No production registry wiring lives here. These plans generate configs for
real A/B runs, then measured results decide whether anything graduates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

Plan = Literal["fast", "balanced_character", "balanced_style", "quality"]


PLAN_PRESETS: dict[Plan, dict[str, Any]] = {
    "fast": {
        "algorithm": "asr_tlora",
        "network_dim_scale": 0.75,
        "caption_dropout_rate": 0.10,
        "min_rank_ratio": 0.50,
        "alpha_rank_scale": 0.80,
        "grad_checkpointing": False,
        "quality_gain_required": 0.00,
        "speed_gain_required": 0.12,
        "max_speed_overhead": 0.00,
    },
    "balanced_character": {
        "algorithm": "dora",
        "network_dim_scale": 1.00,
        "caption_dropout_rate": 0.08,
        "min_rank_ratio": 1.00,
        "alpha_rank_scale": 1.00,
        "grad_checkpointing": False,
        "quality_gain_required": 0.08,
        "speed_gain_required": 0.00,
        "max_speed_overhead": 0.05,
    },
    "balanced_style": {
        "algorithm": "loha",
        "network_dim_scale": 1.00,
        "caption_dropout_rate": 0.18,
        "min_rank_ratio": 1.00,
        "alpha_rank_scale": 1.00,
        "grad_checkpointing": False,
        "quality_gain_required": 0.10,
        "speed_gain_required": 0.00,
        "max_speed_overhead": 0.08,
    },
    "quality": {
        "algorithm": "asr_tlora",
        "network_dim_scale": 1.50,
        "caption_dropout_rate": 0.14,
        "min_rank_ratio": 0.35,
        "alpha_rank_scale": 1.25,
        "grad_checkpointing": True,
        "quality_gain_required": 0.18,
        "speed_gain_required": 0.00,
        "max_speed_overhead": 0.35,
    },
}


def build_experiment_config(
    base: Mapping[str, Any],
    *,
    plan: Plan,
    run_suffix: str = "lora_research_v0",
    max_steps: int | None = None,
    dataset_source: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base))
    preset = PLAN_PRESETS[plan]

    output = cfg.setdefault("output", {})
    base_name = str(output.get("name") or "anima_lora")
    name = f"{base_name}_{run_suffix}_{plan}"
    output["name"] = name
    dataset = cfg.setdefault("dataset", {})
    if dataset_source:
        dataset["source"] = dataset_source

    anima = cfg.setdefault("backend", {}).setdefault("animaLora", {})
    anima["outputName"] = name
    anima["captionDropoutRate"] = preset["caption_dropout_rate"]
    anima["gradientCheckpointing"] = bool(preset["grad_checkpointing"])
    if max_steps is not None:
        cfg.setdefault("schedule", {})["maxSteps"] = int(max_steps)
        anima["maxTrainSteps"] = int(max_steps)
    if smoke:
        cfg.setdefault("sampling", {})["enabled"] = False
        anima["useCmmd"] = False
        anima["validationSplitNum"] = 0
        anima["compileMode"] = None

    base_dim = int(anima.get("networkDim") or cfg.get("network", {}).get("rank") or 16)
    network_dim = max(1, int(round(base_dim * float(preset["network_dim_scale"]))))
    anima["networkDim"] = network_dim
    anima["networkAlpha"] = float(network_dim)

    lora = anima.setdefault("lora", {})
    lora["algorithm"] = preset["algorithm"]
    if preset["algorithm"] == "asr_tlora":
        min_rank = max(1, min(network_dim, math.ceil(network_dim * float(preset["min_rank_ratio"]))))
        lora.update(
            {
                "useTimestepMask": True,
                "perSampleTimestepMask": True,
                "minRank": min_rank,
                "alphaRankScale": preset["alpha_rank_scale"],
            }
        )
    else:
        lora["useTimestepMask"] = False

    dataset.setdefault("caption", {})["dropRate"] = preset["caption_dropout_rate"]
    return cfg


def build_all_experiment_configs(
    base: Mapping[str, Any],
    *,
    run_suffix: str = "lora_research_v0",
    max_steps: int | None = None,
    dataset_source: str | None = None,
    smoke: bool = False,
) -> dict[Plan, dict[str, Any]]:
    return {
        plan: build_experiment_config(
            base,
            plan=plan,
            run_suffix=run_suffix,
            max_steps=max_steps,
            dataset_source=dataset_source,
            smoke=smoke,
        )
        for plan in PLAN_PRESETS
    }


def append_result(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def passes_promotion_gate(
    baseline: Mapping[str, float | int | bool],
    candidate: Mapping[str, float | int | bool | str],
) -> bool:
    plan = str(candidate.get("plan", "balanced_style"))
    preset = PLAN_PRESETS[plan]  # type: ignore[index]
    if candidate.get("nan", False) or candidate.get("black_preview_count", 0):
        return False

    quality_gain = float(candidate.get("quality_score", 0.0)) - float(
        baseline.get("quality_score", 0.0)
    )
    if quality_gain < float(preset["quality_gain_required"]):
        return False

    base_speed = float(baseline.get("seconds_per_step", 0.0))
    cand_speed = float(candidate.get("seconds_per_step", 0.0))
    if base_speed <= 0.0 or cand_speed <= 0.0:
        return True
    speed_gain = (base_speed - cand_speed) / base_speed
    speed_gain_required = float(preset["speed_gain_required"])
    if speed_gain_required > 0.0 and speed_gain < speed_gain_required:
        return False
    return cand_speed <= base_speed * (1.0 + float(preset["max_speed_overhead"]))


def load_json_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a LoRA research experiment config.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", choices=sorted(PLAN_PRESETS))
    parser.add_argument("--all", action="store_true", help="Write one config per research plan.")
    parser.add_argument("--max-steps", type=int, help="Override max steps for smoke tests.")
    parser.add_argument("--dataset-source", help="Override dataset.source in generated configs.")
    parser.add_argument("--smoke", action="store_true", help="Disable sampling/CMMD validation.")
    parser.add_argument("--baseline-result", type=Path, help="Baseline JSON result for promotion gate.")
    parser.add_argument("--candidate-result", type=Path, help="Candidate JSON result for promotion gate.")
    args = parser.parse_args(argv)

    if args.baseline_result or args.candidate_result:
        if not args.baseline_result or not args.candidate_result:
            parser.error("--baseline-result and --candidate-result must be used together")
        ok = passes_promotion_gate(
            load_json_record(args.baseline_result),
            load_json_record(args.candidate_result),
        )
        print("pass" if ok else "fail")
        return 0 if ok else 2

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if args.all:
        args.out.mkdir(parents=True, exist_ok=True)
        for plan, candidate in build_all_experiment_configs(
            base,
            max_steps=args.max_steps,
            dataset_source=args.dataset_source,
            smoke=args.smoke,
        ).items():
            (args.out / f"{plan}.yaml").write_text(
                yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        return 0

    if args.plan is None:
        parser.error("--plan is required unless --all or --*-result is used")
    candidate = build_experiment_config(
        base,
        plan=args.plan,
        max_steps=args.max_steps,
        dataset_source=args.dataset_source,
        smoke=args.smoke,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
