from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from tqdm import tqdm

import Inference_Attack_PoolSize_Representative as base


FIXED_DELTA = 0.15
FIXED_MECHANISM = "EM"

DEFAULT_OPT_COUNTS = (10, 20, 30, 40, 50)
DEFAULT_CLIP_BOUNDS = tuple(round(i / 10.0, 1) for i in range(1, 11))
DEFAULT_EPSILONS = (1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_ATTACK_K = 50

# Final utility-tuned clipping values under delta=0.15.
DEFAULT_OPTCOUNT_CLIP_BY_DATASET = {
    "QWS": 0.4,
    "Normal": 0.2,
}

ATTACK_METRIC_KEYS = (
    "release_pool_size",
    "attack_hypothesis_count",
    "representative_count",
    "random_guess_accuracy_percent",
    "attack_accuracy_percent",
    "chance_normalized_recovery_advantage",
    "constraint_nmae",
    "random_guess_constraint_nmae",
    "true_rank_percentile",
)


ClipKey = Tuple[str, float, float]
OptKey = Tuple[str, float, int]


# =============================================================================
# Helpers
# =============================================================================

def stable_unique_floats(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        value = float(value)
        if not any(abs(value - existing) <= 1e-12 for existing in result):
            result.append(value)
    return result


def choose_epsilons(values: Optional[Sequence[float]]) -> list[float]:
    selected = (
        list(DEFAULT_EPSILONS)
        if values is None
        else stable_unique_floats(values)
    )
    if not selected:
        raise ValueError("At least one epsilon value is required.")
    if any(value <= 0.0 for value in selected):
        raise ValueError("Every epsilon must be positive.")
    return selected


def choose_clip_bounds(values: Optional[Sequence[float]]) -> list[float]:
    selected = (
        list(DEFAULT_CLIP_BOUNDS)
        if values is None
        else stable_unique_floats(values)
    )
    if not selected:
        raise ValueError("At least one ClipBound value is required.")
    if any(not 0.0 < value <= 1.0 for value in selected):
        raise ValueError("Every ClipBound must satisfy 0 < ClipBound <= 1.")
    return selected


def choose_opt_counts(values: Optional[Sequence[int]]) -> list[int]:
    selected = (
        list(DEFAULT_OPT_COUNTS)
        if values is None
        else list(dict.fromkeys(int(value) for value in values))
    )
    if not selected:
        raise ValueError("At least one OptCount value is required.")
    if any(value < 2 for value in selected):
        raise ValueError("Every OptCount must be >= 2.")
    return selected


def choose_distances(values: Optional[Sequence[str]]) -> list[str]:
    if values is None:
        selected = ["hamming"]
    else:
        selected = list(dict.fromkeys(str(value).lower() for value in values))

    unsupported = set(selected) - set(base.SUPPORTED_DISTANCES)
    if unsupported:
        raise ValueError(f"Unsupported distances: {sorted(unsupported)}")
    if not selected:
        raise ValueError("At least one distance is required.")
    return selected


def result_path(dataset: str, cfg) -> Path:
    dataset_safe = str(dataset).strip().replace("/", "_").replace("\\", "_")
    result_dir = Path(__file__).resolve().parent / cfg.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / (
        f"{dataset_safe}_delta015_EM_ClipBound_OptCount_Inference_Ablation.txt"
    )


def atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def fmt(mean_value: float, std_value: float) -> str:
    return f"{float(mean_value):.6f} ± {float(std_value):.6f}"


def metric_cell(stats: base.RunningMetricStats, key: str) -> str:
    mean_value, std_value = stats.mean_std(key)
    return fmt(mean_value, std_value)


def validate_stats_count(
    stats: base.RunningMetricStats,
    completed_runs: int,
    setting,
) -> None:
    if stats.count != completed_runs:
        raise RuntimeError(
            f"Setting {setting} has {stats.count} runs; "
            f"expected {completed_runs}."
        )


def make_delta_para() -> dict:
    para = dict(base.dp_para)
    para["max_utility_degradation"] = float(FIXED_DELTA)
    base.validate_dp_para(para)
    return para


def build_pool(
    source,
    opt_count: int,
    para_delta: dict,
):
    pool = base.build_representative_pool(
        source=source,
        optimal_count=int(opt_count),
        para=para_delta,
    )

    expected_size = int(opt_count) + int(pool.representative_count)
    if len(pool.candidates) != expected_size:
        raise RuntimeError(
            f"Pool-size invariant failed for OptCount={opt_count}: "
            f"len(pool)={len(pool.candidates)} != "
            f"{opt_count}+{pool.representative_count}."
        )
    return pool


# =============================================================================
# Result writer
# =============================================================================

def write_results(
    output_path: Path,
    dataset: str,
    task_count: int,
    completed_runs: int,
    total_runs: int,
    attack_k: int,
    epsilon_values: Sequence[float],
    clip_bounds: Sequence[float],
    opt_counts: Sequence[int],
    optcount_clip_bound: float,
    distances: Sequence[str],
    clip_stats: Dict[ClipKey, base.RunningMetricStats],
    opt_stats: Dict[OptKey, base.RunningMetricStats],
    run_clip: bool,
    run_opt: bool,
) -> None:

    lines = [
        f"Dataset: {dataset}",
        f"Tasks: {int(task_count)}",
        f"TotalPipelineRuns: {int(total_runs)}",
        f"CompletedRuns: {int(completed_runs)}",
        f"Delta: {FIXED_DELTA:.6f}",
        f"Mechanism: {FIXED_MECHANISM}",
        f"FixedAttackHypothesisCountK: {int(attack_k)}",
        f"RandomGuessAccuracy%: {100.0 / float(attack_k):.6f}",
        "EpsilonValues: " + ",".join(f"{float(v):g}" for v in epsilon_values),
        "AttackDistances: " + ",".join(str(v) for v in distances),
        (
            "AttackRule: fixed public hypothesis set K for every ablation setting; "
            "infer argmin_j d(CS*(H_j), CS_obs)"
        ),
        (
            "CandidateSourceReuse: within one task/run, one source built with "
            "max(OptCount) is reused for all ClipBound and OptCount settings"
        ),
        (
            "PrivacyControl: the same evaluator-selected victim requirement and "
            "the same K public attack templates are reused across all settings"
        ),
    ]

    if run_clip:
        lines.extend(
            [
                "",
                "[ClipBoundAblation]",
                f"FixedDelta: {FIXED_DELTA:.6f}",
                "FixedOptCount: 50",
                f"Mechanism: {FIXED_MECHANISM}",
                "ClipBounds: " + ",".join(f"{float(v):g}" for v in clip_bounds),
                "\t".join(
                    [
                        "Distance",
                        "Epsilon",
                        "ClipBound",
                        "ReleasePoolSize(Mean±Std)",
                        "RepresentativeCount(Mean±Std)",
                        "AttackHypothesisCount(Mean±Std)",
                        "RandomGuessAccuracy%(Mean±Std)",
                        "HypothesisRecoveryAccuracy%(Mean±Std)",
                        "ChanceNormalizedRecoveryAdvantage(Mean±Std)",
                        "ConstraintNMAE(Mean±Std)",
                        "RandomGuessConstraintNMAE(Mean±Std)",
                        "TrueRankPercentile(Mean±Std)",
                    ]
                ),
            ]
        )

        for distance_name in distances:
            for epsilon in epsilon_values:
                for clip_bound in clip_bounds:
                    key: ClipKey = (
                        str(distance_name),
                        float(epsilon),
                        float(clip_bound),
                    )
                    stats = clip_stats[key]
                    validate_stats_count(stats, completed_runs, key)
                    lines.append(
                        "\t".join(
                            [
                                str(distance_name),
                                f"{float(epsilon):.6f}",
                                f"{float(clip_bound):.6f}",
                                metric_cell(stats, "release_pool_size"),
                                metric_cell(stats, "representative_count"),
                                metric_cell(stats, "attack_hypothesis_count"),
                                metric_cell(stats, "random_guess_accuracy_percent"),
                                metric_cell(stats, "attack_accuracy_percent"),
                                metric_cell(
                                    stats,
                                    "chance_normalized_recovery_advantage",
                                ),
                                metric_cell(stats, "constraint_nmae"),
                                metric_cell(
                                    stats,
                                    "random_guess_constraint_nmae",
                                ),
                                metric_cell(stats, "true_rank_percentile"),
                            ]
                        )
                    )

    if run_opt:
        lines.extend(
            [
                "",
                "[OptCountAblation]",
                f"FixedDelta: {FIXED_DELTA:.6f}",
                f"FixedClipBound: {float(optcount_clip_bound):.6f}",
                f"Mechanism: {FIXED_MECHANISM}",
                "OptCounts: " + ",".join(str(int(v)) for v in opt_counts),
                "\t".join(
                    [
                        "Distance",
                        "Epsilon",
                        "OptCount",
                        "ClipBound",
                        "ReleasePoolSize(Mean±Std)",
                        "RepresentativeCount(Mean±Std)",
                        "AttackHypothesisCount(Mean±Std)",
                        "RandomGuessAccuracy%(Mean±Std)",
                        "HypothesisRecoveryAccuracy%(Mean±Std)",
                        "ChanceNormalizedRecoveryAdvantage(Mean±Std)",
                        "ConstraintNMAE(Mean±Std)",
                        "RandomGuessConstraintNMAE(Mean±Std)",
                        "TrueRankPercentile(Mean±Std)",
                    ]
                ),
            ]
        )

        for distance_name in distances:
            for epsilon in epsilon_values:
                for opt_count in opt_counts:
                    key: OptKey = (
                        str(distance_name),
                        float(epsilon),
                        int(opt_count),
                    )
                    stats = opt_stats[key]
                    validate_stats_count(stats, completed_runs, key)
                    lines.append(
                        "\t".join(
                            [
                                str(distance_name),
                                f"{float(epsilon):.6f}",
                                str(int(opt_count)),
                                f"{float(optcount_clip_bound):.6f}",
                                metric_cell(stats, "release_pool_size"),
                                metric_cell(stats, "representative_count"),
                                metric_cell(stats, "attack_hypothesis_count"),
                                metric_cell(stats, "random_guess_accuracy_percent"),
                                metric_cell(stats, "attack_accuracy_percent"),
                                metric_cell(
                                    stats,
                                    "chance_normalized_recovery_advantage",
                                ),
                                metric_cell(stats, "constraint_nmae"),
                                metric_cell(
                                    stats,
                                    "random_guess_constraint_nmae",
                                ),
                                metric_cell(stats, "true_rank_percentile"),
                            ]
                        )
                    )

    atomic_write_text(output_path, "\n".join(lines) + "\n")


# =============================================================================
# Main ablation experiment
# =============================================================================

def run_ablation(
    dataset: str,
    workers: int,
    task_limit: Optional[int],
    runs_override: Optional[int],
    rebuild_cache: bool,
    epsilon_filter: Optional[Sequence[float]],
    clip_filter: Optional[Sequence[float]],
    opt_count_filter: Optional[Sequence[int]],
    distance_filter: Optional[Sequence[str]],
    attack_k: int,
    optcount_clip_override: Optional[float],
    ablation_mode: str,
) -> None:

    base.validate_dp_para(base.dp_para)
    cfg = base.load_config(dataset=dataset)

    if not cfg.test_only:
        raise RuntimeError(
            "This attack requires TEST_ONLY=True so the first 75% training split "
            "is attacker background and held-out workflows are evaluation contexts."
        )

    attack_k = int(attack_k)
    if attack_k < 2:
        raise ValueError("attack-k must be >= 2.")

    epsilon_values = choose_epsilons(epsilon_filter)
    clip_bounds = choose_clip_bounds(clip_filter)
    opt_counts = choose_opt_counts(opt_count_filter)
    distances = choose_distances(distance_filter)

    run_clip = ablation_mode in {"both", "clip"}
    run_opt = ablation_mode in {"both", "optcount"}

    if not run_clip and not run_opt:
        raise ValueError(f"Unsupported ablation mode: {ablation_mode}")


    fixed_clip_opt_count = 50

    if run_opt:
        max_opt_count = max(max(opt_counts), fixed_clip_opt_count if run_clip else 0)
    else:
        max_opt_count = fixed_clip_opt_count

    optcount_clip_bound = (
        float(DEFAULT_OPTCOUNT_CLIP_BY_DATASET[dataset])
        if optcount_clip_override is None
        else float(optcount_clip_override)
    )
    if not 0.0 < optcount_clip_bound <= 1.0:
        raise ValueError("OptCount-ablation fixed ClipBound must be in (0, 1].")

    pipeline_runs = (
        int(base.dp_para.get("pipeline_runs", 20))
        if runs_override is None
        else int(runs_override)
    )
    if pipeline_runs <= 0:
        raise ValueError("pipeline run count must be positive.")

    proxy_pool_size = base.dataset_proxy_pool_size(base.dp_para, dataset)
    if max_opt_count > proxy_pool_size:
        raise ValueError(
            f"max OptCount={max_opt_count} exceeds proxy_pool_size={proxy_pool_size}."
        )

    all_tasks = base.load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=cfg.test_only,
        min_candidates=1,
    )

    target_tasks = base.target_tasks_from_config(
        all_tasks=all_tasks,
        cfg=cfg,
        task_limit=task_limit,
    )
    if not target_tasks:
        raise RuntimeError("No held-out workflow context is available.")

    repository = base.load_public_service_repository(
        dataset=dataset,
        node_filename=cfg.node_file,
        service_filename=cfg.service_file,
    )

    training_bank = base.load_training_requirement_bank(
        dataset=dataset,
        node_filename=cfg.node_file,
        service_filename=cfg.service_file,
        repository=repository,
    )

    needed_categories = {
        int(constraint.category_id)
        for task in target_tasks
        for constraint in task.local_constraints
    }
    missing_categories = sorted(
        needed_categories - set(training_bank.samples_by_category)
    )
    if missing_categories:
        raise RuntimeError(
            "Training split lacks local requirement observations for categories: "
            f"{missing_categories}"
        )

    para_delta = make_delta_para()

    precompute_start = time.time()

    (
        attack_database,
        attack_cache,
        attack_cache_metadata,
        cache_hit,
    ) = base.build_attack_database(
        dataset=dataset,
        tasks=target_tasks,
        training_bank=training_bank,
        repository=repository,
        base_count=attack_k,
        cfg=cfg,
        workers=workers,
        rebuild_cache=rebuild_cache,
    )

    print("============================================================")
    print("SEQUENCE-DP PRIVACY PARAMETER ABLATION")
    print("============================================================")
    print(f"Dataset                    : {dataset}")
    print(f"Tasks                      : {len(target_tasks)}")
    print(f"Pipeline runs              : {pipeline_runs}")
    print(f"Delta                      : {FIXED_DELTA}")
    print(f"Mechanism                  : {FIXED_MECHANISM}")
    print(f"Fixed attack K             : {attack_k}")
    print(f"Random guess accuracy      : {100.0 / attack_k:.6f}%")
    print(f"Epsilons                   : {epsilon_values}")
    print(f"Attack distances           : {distances}")
    print(f"Proxy source size          : {proxy_pool_size}")

    if run_clip:
        print(
            "ClipBound ablation         : "
            f"OptCount=50, ClipBounds={clip_bounds}"
        )

    if run_opt:
        print(
            "OptCount ablation          : "
            f"OptCounts={opt_counts}, fixed ClipBound={optcount_clip_bound}"
        )

    print(
        "Attack cache                : "
        f"{attack_cache} ({'HIT' if cache_hit else 'BUILT/UPDATED'})"
    )
    print(
        "Attack-bank elapsed         : "
        f"{time.time() - precompute_start:.2f}s"
    )
    print("============================================================\n")

    output_path = result_path(dataset, cfg)

    clip_stats: Dict[ClipKey, base.RunningMetricStats] = {}
    opt_stats: Dict[OptKey, base.RunningMetricStats] = {}

    total_start = time.time()

    for run_index in range(1, pipeline_runs + 1):
        run_start = time.time()

        run_clip_acc: Dict[ClipKey, base.MetricAccumulator] = {}
        run_opt_acc: Dict[OptKey, base.MetricAccumulator] = {}

        for base_task in tqdm(
            target_tasks,
            desc=f"Ablation {run_index}/{pipeline_runs}",
        ):
            task_index = int(base_task.task_index)
            attack_model = attack_database[task_index]

            # Same evaluator-only private requirement for every ablation setting.
            victim_task = base.make_hypothetical_task(
                base_task=base_task,
                hypothesis=attack_model.true_hypothesis,
                repository=repository,
            )

            source_seed = (
                int(cfg.ga_seed)
                + int(run_index) * 10_000_019
                + task_index
            )

            source = base.build_candidate_source(
                task=victim_task,
                cfg=cfg,
                seed=source_seed,
                optimal_count=max_opt_count,
                proxy_pool_size=proxy_pool_size,
            )

            if len(source.quality_candidates) < max_opt_count:
                raise RuntimeError(
                    f"Task {task_index}: source contains only "
                    f"{len(source.quality_candidates)} quality candidates, "
                    f"but max OptCount={max_opt_count} is required."
                )

            if run_clip:
                clip_pool = build_pool(
                    source=source,
                    opt_count=fixed_clip_opt_count,
                    para_delta=para_delta,
                )
                clip_rep_count = int(clip_pool.representative_count)
                clip_pool_size = len(clip_pool.candidates)

                for epsilon in epsilon_values:
                    for clip_bound in clip_bounds:
                        selected = base.select_candidate(
                            pool=clip_pool,
                            mechanism=FIXED_MECHANISM,
                            epsilon=float(epsilon),
                            clip_bound=float(clip_bound),
                        )

                        observation = base.selected_candidate_observation(
                            selected,
                            victim_task,
                        )

                        for distance_name in distances:
                            key: ClipKey = (
                                str(distance_name),
                                float(epsilon),
                                float(clip_bound),
                            )

                            metrics = base.attack_observation(
                                attack_model=attack_model,
                                observation=observation,
                                hypothesis_count=attack_k,
                                distance_name=distance_name,
                                release_pool_size=clip_pool_size,
                                representative_count=clip_rep_count,
                            )

                            run_clip_acc.setdefault(
                                key,
                                base.MetricAccumulator(ATTACK_METRIC_KEYS),
                            ).add(metrics)

            if run_opt:
                opt_pools = {
                    int(opt_count): build_pool(
                        source=source,
                        opt_count=int(opt_count),
                        para_delta=para_delta,
                    )
                    for opt_count in opt_counts
                }

                for opt_count in opt_counts:
                    pool = opt_pools[int(opt_count)]
                    rep_count = int(pool.representative_count)
                    pool_size = len(pool.candidates)

                    for epsilon in epsilon_values:
                        selected = base.select_candidate(
                            pool=pool,
                            mechanism=FIXED_MECHANISM,
                            epsilon=float(epsilon),
                            clip_bound=float(optcount_clip_bound),
                        )

                        observation = base.selected_candidate_observation(
                            selected,
                            victim_task,
                        )

                        for distance_name in distances:
                            key: OptKey = (
                                str(distance_name),
                                float(epsilon),
                                int(opt_count),
                            )

                            metrics = base.attack_observation(
                                attack_model=attack_model,
                                observation=observation,
                                hypothesis_count=attack_k,
                                distance_name=distance_name,
                                release_pool_size=pool_size,
                                representative_count=rep_count,
                            )

                            run_opt_acc.setdefault(
                                key,
                                base.MetricAccumulator(ATTACK_METRIC_KEYS),
                            ).add(metrics)

        if run_clip:
            expected_clip_settings = (
                len(distances)
                * len(epsilon_values)
                * len(clip_bounds)
            )
            if len(run_clip_acc) != expected_clip_settings:
                raise RuntimeError(
                    f"ClipBound ablation produced {len(run_clip_acc)} settings; "
                    f"expected {expected_clip_settings}."
                )

            for key, accumulator in run_clip_acc.items():
                clip_stats.setdefault(
                    key,
                    base.RunningMetricStats(ATTACK_METRIC_KEYS),
                ).update(accumulator.mean())

        if run_opt:
            expected_opt_settings = (
                len(distances)
                * len(epsilon_values)
                * len(opt_counts)
            )
            if len(run_opt_acc) != expected_opt_settings:
                raise RuntimeError(
                    f"OptCount ablation produced {len(run_opt_acc)} settings; "
                    f"expected {expected_opt_settings}."
                )

            for key, accumulator in run_opt_acc.items():
                opt_stats.setdefault(
                    key,
                    base.RunningMetricStats(ATTACK_METRIC_KEYS),
                ).update(accumulator.mean())

        write_results(
            output_path=output_path,
            dataset=dataset,
            task_count=len(target_tasks),
            completed_runs=run_index,
            total_runs=pipeline_runs,
            attack_k=attack_k,
            epsilon_values=epsilon_values,
            clip_bounds=clip_bounds,
            opt_counts=opt_counts,
            optcount_clip_bound=optcount_clip_bound,
            distances=distances,
            clip_stats=clip_stats,
            opt_stats=opt_stats,
            run_clip=run_clip,
            run_opt=run_opt,
        )

        # Compact progress at epsilon=3 when available.
        progress_parts = []

        if 3.0 in epsilon_values and run_clip:
            reference_clip = (
                0.4 if dataset == "QWS" else 0.2
            )
            if reference_clip in clip_bounds:
                key = ("hamming", 3.0, float(reference_clip))
                if key in clip_stats:
                    hra, _ = clip_stats[key].mean_std(
                        "attack_accuracy_percent"
                    )
                    cnra, _ = clip_stats[key].mean_std(
                        "chance_normalized_recovery_advantage"
                    )
                    progress_parts.append(
                        f"Clip@{reference_clip:g}: "
                        f"HRA={hra:.2f}% CNRA={cnra:.4f}"
                    )

        if 3.0 in epsilon_values and run_opt and 50 in opt_counts:
            key = ("hamming", 3.0, 50)
            if key in opt_stats:
                hra, _ = opt_stats[key].mean_std(
                    "attack_accuracy_percent"
                )
                cnra, _ = opt_stats[key].mean_std(
                    "chance_normalized_recovery_advantage"
                )
                progress_parts.append(
                    f"Opt50: HRA={hra:.2f}% CNRA={cnra:.4f}"
                )

        print(
            f"Run {run_index}/{pipeline_runs} finished"
            + (
                " | " + " | ".join(progress_parts)
                if progress_parts
                else ""
            )
        )
        print(
            f"saved={output_path} | "
            f"run_elapsed={time.time() - run_start:.2f}s | "
            f"total_elapsed={time.time() - total_start:.2f}s"
        )

    print("\nAll ClipBound/OptCount inference-ablation results saved to:")
    print(output_path)
    print(f"Total elapsed: {time.time() - total_start:.2f}s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Sequence-DP inference-attack ablation for ClipBound "
            "and OptCount. Fixed delta=0.15 and mechanism=EM."
        )
    )

    parser.add_argument(
        "dataset",
        choices=["QWS", "Normal"],
    )

    parser.add_argument(
        "--ablation",
        choices=["both", "clip", "optcount"],
        default="both",
        help=(
            "Run both ablations, only ClipBound, or only OptCount. "
            "Default: both."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Processes used only for the fixed-K public attack-bank "
            "precomputation. Default: 4."
        ),
    )

    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional smoke-test task limit. Omit for all held-out tasks.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=(
            "Optional pipeline-run override. Omit for "
            "dp_para['pipeline_runs'] (or 20 if absent)."
        ),
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuilding the fixed-K public attack-template cache.",
    )

    parser.add_argument(
        "--attack-k",
        type=int,
        default=DEFAULT_ATTACK_K,
        help=(
            "Fixed public attack hypothesis count used for EVERY setting. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=None,
        help="Default: 1 2 3 4 5.",
    )

    parser.add_argument(
        "--clip-bounds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "ClipBound values for the ClipBound ablation. "
            "Default: 0.1 0.2 ... 1.0."
        ),
    )

    parser.add_argument(
        "--opt-counts",
        nargs="+",
        type=int,
        default=None,
        help="OptCount values. Default: 10 20 30 40 50.",
    )

    parser.add_argument(
        "--optcount-clip-bound",
        type=float,
        default=None,
        help=(
            "Fixed ClipBound for the OptCount ablation. "
            "Default: QWS=0.4, Normal=0.2."
        ),
    )

    parser.add_argument(
        "--distances",
        nargs="+",
        choices=list(base.SUPPORTED_DISTANCES),
        default=None,
        help=(
            "Attack distances. Default: hamming. "
            "Use 'hamming jaccard qos' for supplementary robustness."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_ablation(
        dataset=args.dataset,
        workers=args.workers,
        task_limit=args.task_limit,
        runs_override=args.runs,
        rebuild_cache=args.rebuild_cache,
        epsilon_filter=args.epsilons,
        clip_filter=args.clip_bounds,
        opt_count_filter=args.opt_counts,
        distance_filter=args.distances,
        attack_k=args.attack_k,
        optcount_clip_override=args.optcount_clip_bound,
        ablation_mode=args.ablation,
    )


if __name__ == "__main__":
    main()
