from __future__ import annotations

import json
import math
import statistics
import sys
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from config_loader import REFERENCE_FORMAT, load_config
from dp_perturb import dataset_dp_settings, dp_para, validate_dp_para
from dp_qosc import (
    build_candidate_source,
    build_representative_pool,
    select_candidate,
    selection_probability_grid,
)
from loadData_GA import SLATask, load_sla_tasks, resolve_dataset_paths
from sla_core import (
    SLAOnDemandGA,
    boundary_optimality,
    distance_gap,
    file_sha256,
    load_exact_reference,
)


SettingKey = Tuple[str, Optional[float], Optional[float]]


METRIC_KEYS = (
    "normalized_sequence_deviation",
    "near_optimal_probability",
    "optimality_percent",
    "qsd",
    "mean_quantile_deviation",
    "max_quantile_deviation",
)


@dataclass
class MetricAccumulator:
    sums: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in METRIC_KEYS}
    )
    count: int = 0

    def add(self, metrics: Dict[str, float]) -> None:
        for key in METRIC_KEYS:
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"Metric {key} is not finite: {value}")
            self.sums[key] += value

        self.count += 1

    def mean(self) -> Dict[str, float]:
        if self.count == 0:
            raise RuntimeError("Cannot average an empty metric accumulator.")

        return {
            key: self.sums[key] / self.count
            for key in METRIC_KEYS
        }


@dataclass
class RunningMetricStats:


    count: int = 0
    means: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in METRIC_KEYS}
    )
    m2: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in METRIC_KEYS}
    )

    def update(self, metrics: Dict[str, float]) -> None:
        new_count = self.count + 1
        for key in METRIC_KEYS:
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"Metric {key} is not finite: {value}")

            delta = value - self.means[key]
            self.means[key] += delta / new_count
            delta2 = value - self.means[key]
            self.m2[key] += delta * delta2

        self.count = new_count

    def mean_std(self, key: str) -> Tuple[float, float]:
        if self.count <= 0:
            raise RuntimeError("Running statistics are empty.")

        mean_value = float(self.means[key])
        if self.count == 1:
            return mean_value, 0.0

        variance = self.m2[key] / (self.count - 1)
        return mean_value, math.sqrt(max(0.0, variance))



@dataclass
class ScalarAccumulator:

    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Representative sequence count must be finite and non-negative: {value}"
            )
        self.total += value
        self.count += 1

    def mean(self) -> float:
        if self.count <= 0:
            raise RuntimeError("Scalar accumulator is empty.")
        return float(self.total / self.count)


@dataclass
class RunningScalarStats:

    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Running scalar value is not finite: {value}")

        new_count = self.count + 1
        delta = value - self.mean_value
        self.mean_value += delta / new_count
        delta2 = value - self.mean_value
        self.m2 += delta * delta2
        self.count = new_count

    def mean_std(self) -> Tuple[float, float]:
        if self.count <= 0:
            raise RuntimeError("Running scalar statistics are empty.")
        if self.count == 1:
            return float(self.mean_value), 0.0
        variance = self.m2 / (self.count - 1)
        return float(self.mean_value), math.sqrt(max(0.0, variance))


def parse_arguments() -> Tuple[str, str]:
    if len(sys.argv) not in {2, 3}:
        print(
            "Usage: python OnDemand_GA.py <dataset> [main|ablation|delta|optcount]\n"
            "Examples:\n"
            "  python OnDemand_GA.py QWS\n"
            "  python OnDemand_GA.py Normal ablation\n"
            "  python OnDemand_GA.py Normal delta\n"
            "  python OnDemand_GA.py Normal optcount"
        )
        raise SystemExit(2)

    dataset = sys.argv[1]
    mode = "main" if len(sys.argv) == 2 else sys.argv[2].strip().lower()

    if mode in {
        "ablation",
        "delta",
        "delta_ablation",
        "optcount",
        "optcount_ablation",
        "opt_count",
        "optimal_count",
    }:
        mode = "ablation"

    if mode not in {"main", "ablation"}:
        raise ValueError(
            "Experiment mode must be 'main' or joint 'ablation' "
            "(aliases: delta, optcount)."
        )

    return dataset, mode

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")

    if len(values) == 1:
        return float(values[0]), 0.0

    return (
        float(statistics.mean(values)),
        float(statistics.stdev(values)),
    )


def target_tasks_from_config(
    all_tasks: List[SLATask],
    cfg,
) -> List[SLATask]:
    tasks = list(all_tasks)

    if cfg.exact_max_tasks is not None:
        tasks = tasks[: cfg.exact_max_tasks]

    return tasks


def verify_task_record(
    record: Dict,
    task: SLATask,
    tolerance: float = 1e-12,
) -> None:
    expected_category_ids = [
        int(constraint.category_id)
        for constraint in task.local_constraints
    ]
    expected_candidate_counts = [
        len(category)
        for category in task.services
    ]
    expected_q2_intervals = [
        [float(constraint.q2[0]), float(constraint.q2[1])]
        for constraint in task.local_constraints
    ]
    expected_q3_intervals = [
        [float(constraint.q3[0]), float(constraint.q3[1])]
        for constraint in task.local_constraints
    ]

    checks = [
        (
            int(record["task_index"]) == int(task.task_index),
            "task_index",
        ),
        (
            int(record["length"]) == len(task.services),
            "task length",
        ),
        (
            list(record["category_ids"]) == expected_category_ids,
            "category IDs",
        ),
        (
            list(record["candidate_counts"])
            == expected_candidate_counts,
            "candidate counts",
        ),
        (
            list(record["local_q2_intervals"])
            == expected_q2_intervals,
            "local q2 intervals",
        ),
        (
            list(record["local_q3_intervals"])
            == expected_q3_intervals,
            "local q3 intervals",
        ),
        (
            abs(
                float(record["global_q2_upper"])
                - float(task.global_q2[1])
            )
            <= tolerance,
            "global q2 upper",
        ),
        (
            abs(
                float(record["global_q3_upper"])
                - float(task.global_q3[1])
            )
            <= tolerance,
            "global q3 upper",
        ),
    ]

    failed = [name for ok, name in checks if not ok]

    if failed:
        raise RuntimeError(
            f"Exact record for task {task.task_index} does not match "
            f"the current task: {failed}"
        )


def verify_reference(
    reference: Dict,
    dataset: str,
    cfg,
    all_tasks: List[SLATask],
):
    node_path, service_path = resolve_dataset_paths(
        dataset,
        cfg.node_file,
        cfg.service_file,
    )
    meta = reference["meta"]
    target_tasks = target_tasks_from_config(all_tasks, cfg)
    expected_target_ids = [
        int(task.task_index)
        for task in target_tasks
    ]

    checks = [
        (
            reference.get("format") == REFERENCE_FORMAT,
            "reference format",
        ),
        (
            meta.get("dataset") == dataset,
            "dataset",
        ),
        (
            meta.get("node_file") == cfg.node_file,
            "node_file",
        ),
        (
            meta.get("service_file") == cfg.service_file,
            "service_file",
        ),
        (
            meta.get("node_sha256") == file_sha256(node_path),
            "node SHA-256",
        ),
        (
            meta.get("service_sha256") == file_sha256(service_path),
            "service SHA-256",
        ),
        (
            bool(meta.get("test_only")) == bool(cfg.test_only),
            "test_only",
        ),
        (
            meta.get("exact_max_tasks") == cfg.exact_max_tasks,
            "EXACT.max_tasks",
        ),
        (
            list(meta.get("target_task_indices", []))
            == expected_target_ids,
            "target task set/order",
        ),
    ]

    failed = [name for ok, name in checks if not ok]

    if failed:
        raise RuntimeError(
            "Exact reference does not match the current experiment.\n"
            f"Mismatched fields: {failed}"
        )

    records = list(reference["tasks"])
    stored_ids = [int(record["task_index"]) for record in records]

    if stored_ids != expected_target_ids:
        raise RuntimeError(
            "Exact reference is incomplete or does not follow "
            "the current target task order."
        )

    task_map = {
        int(task.task_index): task
        for task in target_tasks
    }

    for record in records:
        task_index = int(record["task_index"])
        verify_task_record(record, task_map[task_index])

    return target_tasks, records


def build_ga(
    task: SLATask,
    cfg,
    ga_seed: int,
) -> SLAOnDemandGA:
    return SLAOnDemandGA(
        task,
        population_size=cfg.ga_population,
        max_generations=cfg.ga_max_generations,
        stagnation_patience=cfg.ga_stagnation_patience,
        crossover_rate=cfg.ga_crossover_rate,
        mutation_probability=cfg.ga_mutation_probability,
        elite_count=cfg.ga_elite_count,
        seed=ga_seed,
    )




def build_public_qos_ranges(
    service_feature: Dict,
) -> Dict[int, Tuple[float, float, float, float]]:
    ranges: Dict[int, Tuple[float, float, float, float]] = {}

    for category_key, raw_services in service_feature.items():
        if not raw_services:
            continue

        q2_values = [float(service[-2]) for service in raw_services]
        q3_values = [float(service[-1]) for service in raw_services]
        ranges[int(category_key)] = (
            min(q2_values),
            max(q2_values),
            min(q3_values),
            max(q3_values),
        )

    return ranges


def normalized_qos_deviation(
    selected_value: float,
    exact_value: float,
    lower: float,
    upper: float,
) -> float:

    selected_value = float(selected_value)
    exact_value = float(exact_value)
    lower = float(lower)
    upper = float(upper)

    span = upper - lower
    difference = abs(selected_value - exact_value)

    if span <= 1e-15:
        if difference <= 1e-12:
            return 0.0
        raise RuntimeError(
            "A zero-width public QoS range contains inconsistent values: "
            f"selected={selected_value}, exact={exact_value}, "
            f"range=[{lower}, {upper}]."
        )

    value = difference / span
    if value > 1.0 + 1e-9:
        raise RuntimeError(
            "Normalized QoS deviation exceeds 1 although both services should "
            "belong to the same public category: "
            f"selected={selected_value}, exact={exact_value}, "
            f"range=[{lower}, {upper}], normalized={value}."
        )

    return min(1.0, max(0.0, value))


def normalized_sequence_deviation(
    selected,
    task: SLATask,
    exact_record: Dict,
    service_feature: Dict,
    public_qos_ranges: Dict[int, Tuple[float, float, float, float]],
) -> float:

    exact_indices = [
        int(index)
        for index in exact_record["exact_sequence_concrete_indices"]
    ]

    if len(selected.chromosome) != len(task.services):
        raise RuntimeError(
            f"Task {task.task_index}: selected sequence length mismatch."
        )
    if len(exact_indices) != len(task.services):
        raise RuntimeError(
            f"Task {task.task_index}: Exact sequence length mismatch."
        )

    normalized_deviations: List[float] = []

    for position, (gene, exact_index, constraint) in enumerate(
        zip(
            selected.chromosome,
            exact_indices,
            task.local_constraints,
        )
    ):
        category_id = int(constraint.category_id)
        selected_service = task.services[position][int(gene)]

        if int(selected_service[4]) != category_id:
            raise RuntimeError(
                f"Task {task.task_index}, position {position}: "
                "selected service category mismatch."
            )

        raw_services = service_feature[str(category_id)]
        if not 0 <= exact_index < len(raw_services):
            raise RuntimeError(
                f"Task {task.task_index}, position {position}: "
                f"Exact concrete index {exact_index} is outside public category "
                f"{category_id}."
            )

        if category_id not in public_qos_ranges:
            raise RuntimeError(
                f"Missing public QoS range for category {category_id}."
            )

        exact_service = raw_services[exact_index]
        q2_min, q2_max, q3_min, q3_max = public_qos_ranges[category_id]

        normalized_deviations.append(
            normalized_qos_deviation(
                selected_value=float(selected_service[2]),
                exact_value=float(exact_service[-2]),
                lower=q2_min,
                upper=q2_max,
            )
        )
        normalized_deviations.append(
            normalized_qos_deviation(
                selected_value=float(selected_service[3]),
                exact_value=float(exact_service[-1]),
                lower=q3_min,
                upper=q3_max,
            )
        )

    if not normalized_deviations:
        raise RuntimeError(
            f"Task {task.task_index}: NSD has no QoS component to evaluate."
        )

    return float(statistics.mean(normalized_deviations))



def build_public_quantile_reference(
    service_feature: Dict,
) -> Dict[int, Dict[int, List[float]]]:
    reference: Dict[int, Dict[int, List[float]]] = {}

    for raw_category_id, raw_services in service_feature.items():
        category_id = int(raw_category_id)
        if not raw_services:
            raise RuntimeError(
                f"Public service category {category_id} is empty."
            )

        reference[category_id] = {
            2: sorted(float(service[-2]) for service in raw_services),
            3: sorted(float(service[-1]) for service in raw_services),
        }

    return reference


def empirical_midrank(
    value: float,
    sorted_values: List[float],
    *,
    category_id: int,
    qos_index: int,
) -> float:
    if not sorted_values:
        raise ValueError("Quantile reference distribution cannot be empty.")

    value = float(value)
    left = bisect_left(sorted_values, value)
    right = bisect_right(sorted_values, value)

    if left == right:
        raise RuntimeError(
            "QoS value is absent from its complete public-category reference: "
            f"category={category_id}, q{qos_index}={value}."
        )

    return float((left + 0.5 * (right - left)) / len(sorted_values))


def quantile_sequence_deviation_metrics(
    selected,
    task: SLATask,
    exact_record: Dict,
    service_feature: Dict,
    quantile_reference: Dict[int, Dict[int, List[float]]],
) -> Dict[str, float]:

    chromosome = tuple(int(gene) for gene in selected.chromosome)
    length = len(task.services)
    if len(chromosome) != length:
        raise RuntimeError(
            f"Task {task.task_index}: selected chromosome length mismatch."
        )

    exact_indices = [
        int(index)
        for index in exact_record["exact_sequence_concrete_indices"]
    ]
    if len(exact_indices) != length:
        raise RuntimeError(
            f"Task {task.task_index}: Exact sequence length mismatch."
        )

    deviations: List[float] = []

    for position, (gene, exact_index, constraint) in enumerate(
        zip(chromosome, exact_indices, task.local_constraints)
    ):
        category_id = int(constraint.category_id)

        if not 0 <= gene < len(task.services[position]):
            raise RuntimeError(
                f"Task {task.task_index}, position {position}: "
                f"selected gene {gene} is outside the local candidate set."
            )

        selected_service = task.services[position][gene]
        if int(selected_service[4]) != category_id:
            raise RuntimeError(
                f"Task {task.task_index}, position {position}: "
                "selected service category mismatch."
            )

        category_key = str(category_id)
        if category_key not in service_feature:
            raise RuntimeError(
                f"Public service category {category_id} is missing."
            )
        raw_category = service_feature[category_key]
        if not 0 <= exact_index < len(raw_category):
            raise RuntimeError(
                f"Task {task.task_index}, position {position}: Exact service "
                f"index {exact_index} is outside public category {category_id}."
            )

        exact_service = raw_category[exact_index]
        category_reference = quantile_reference.get(category_id)
        if category_reference is None:
            raise RuntimeError(
                f"Quantile reference is missing category {category_id}."
            )

        for qos_index, selected_value, exact_value in (
            (2, float(selected_service[2]), float(exact_service[-2])),
            (3, float(selected_service[3]), float(exact_service[-1])),
        ):
            selected_rank = empirical_midrank(
                selected_value,
                category_reference[qos_index],
                category_id=category_id,
                qos_index=qos_index,
            )
            exact_rank = empirical_midrank(
                exact_value,
                category_reference[qos_index],
                category_id=category_id,
                qos_index=qos_index,
            )
            displacement = abs(selected_rank - exact_rank)
            if (
                not math.isfinite(displacement)
                or displacement < -1e-12
                or displacement > 1.0 + 1e-12
            ):
                raise RuntimeError(
                    f"Task {task.task_index}: invalid quantile displacement "
                    f"for q{qos_index}: {displacement}"
                )
            deviations.append(
                min(1.0, max(0.0, float(displacement)))
            )

    if not deviations:
        raise RuntimeError(
            f"Task {task.task_index}: quantile deviation has no QoS components."
        )

    mean_qd = float(statistics.mean(deviations))
    qsd = float(
        math.sqrt(statistics.mean(value * value for value in deviations))
    )
    max_qd = float(max(deviations))

    return {
        "qsd": qsd,
        "mean_quantile_deviation": mean_qd,
        "max_quantile_deviation": max_qd,
    }

def exact_sequence_utility(exact_record: Dict) -> float:
    """
    Raw, unclipped Exact utility shared by Sequence-DP and the cross-method
    baselines:

        U(Exact) = [Q2(Exact) + Q3(Exact)] / 2.
    """
    utility = 0.5 * (
        float(exact_record["exact_q2_product"])
        + float(exact_record["exact_q3_product"])
    )
    if not 0.0 <= utility <= 1.0 + 1e-12:
        raise RuntimeError(
            f"Exact utility is outside [0, 1]: {utility}"
        )
    return min(1.0, max(0.0, float(utility)))


def near_optimal_probability_grid(
    pool,
    mechanisms: List[str],
    epsilon_values: List[float],
    clip_bounds: List[float],
    near_optimal_fraction: float,
) -> Dict[SettingKey, float]:
    """
    Compute the ORIGINAL distribution-level NOP@alpha metric.

    Candidate near-optimality is defined from RAW, UNCLIPPED utility inside
    the CURRENT candidate pool:

        e_j = (u_max - u_j) / (u_max - u_min).

    Candidate j is near-optimal when e_j <= alpha.

    The selection probabilities P_j are induced by the configured DP
    mechanism (EM/PNF), epsilon and clipping threshold.  NOP is the total
    probability mass of all near-optimal candidates:

        NOP = 100 * sum_{j: e_j <= alpha} P_j.

    QWS uses alpha=0.05 and Normal uses alpha=0.10 through the existing
    dataset-specific configuration.

    This deterministic PMF calculation consumes no Phase-3 random numbers.
    """
    near_optimal_fraction = float(near_optimal_fraction)
    if not 0.0 < near_optimal_fraction < 1.0:
        raise ValueError(
            "near_optimal_fraction must satisfy 0 < alpha < 1."
        )

    utilities = [
        float(candidate.utility)
        for candidate in pool.candidates
    ]
    if not utilities:
        raise ValueError("Candidate pool cannot be empty.")

    utility_max = max(utilities)
    utility_min = min(utilities)
    utility_spread = utility_max - utility_min

    if utility_spread <= 1e-15:
        near_mask = [True] * len(utilities)
    else:
        near_mask = [
            (utility_max - utility) / utility_spread
            <= near_optimal_fraction + 1e-15
            for utility in utilities
        ]

    probability_grid = selection_probability_grid(
        candidates=pool.candidates,
        mechanisms=mechanisms,
        epsilon_values=epsilon_values,
        clip_bounds=clip_bounds,
    )

    result: Dict[SettingKey, float] = {}

    for setting, probabilities in probability_grid.items():
        if len(probabilities) != len(near_mask):
            raise RuntimeError(
                f"Selection PMF size mismatch for setting {setting}."
            )

        probability = 100.0 * sum(
            float(value)
            for value, is_near in zip(probabilities, near_mask)
            if is_near
        )

        result[setting] = min(
            100.0,
            max(0.0, probability),
        )

    return result


def evaluate_setting_metrics(
    selected,
    task: SLATask,
    exact_record: Dict,
    service_feature: Dict,
    public_qos_ranges: Dict[int, Tuple[float, float, float, float]],
    quantile_reference: Dict[int, Dict[int, List[float]]],
    near_optimal_probability: float,
) -> Dict[str, float]:
    """
    Evaluate one actually released candidate.

    OPT is the SLA-style boundary optimality ratio relative to the full Exact
    solution of the same original task:

        OPT = 100 * D* / D_selected

    with the zero-distance cases handled by boundary_optimality().  Sequence-DP
    candidates are constructed only from the original locally feasible service
    sets and only globally feasible compositions may enter Psi, so no noisy-to-
    original remapping is needed here.
    """
    quantile_metrics = quantile_sequence_deviation_metrics(
        selected=selected,
        task=task,
        exact_record=exact_record,
        service_feature=service_feature,
        quantile_reference=quantile_reference,
    )

    if selected.evaluation.violations != 0:
        raise RuntimeError(
            f"Task {task.task_index}: a globally infeasible candidate was "
            "released from the Sequence-DP candidate pool."
        )

    optimality_percent = 100.0 * boundary_optimality(
        float(exact_record["exact_distance"]),
        float(selected.evaluation.distance),
    )

    return {
        "normalized_sequence_deviation": normalized_sequence_deviation(
            selected=selected,
            task=task,
            exact_record=exact_record,
            service_feature=service_feature,
            public_qos_ranges=public_qos_ranges,
        ),
        "near_optimal_probability": float(near_optimal_probability),
        "optimality_percent": float(optimality_percent),
        **quantile_metrics,
    }

def dp_result_path(dataset: str, cfg) -> Path:
    dataset_safe = (
        str(dataset)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
    )
    result_dir = Path(__file__).resolve().parent / cfg.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{dataset_safe}_dp_qosc_results.txt"


def _atomic_write_text(output_path: Path, text: str) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(output_path)



def write_dp_results(
    output_path: Path,
    dataset: str,
    running_stats: Dict[SettingKey, RunningMetricStats],
    completed_runs: int,
    total_runs: int,
    task_count: int,
    optimal_candidate_count: int,
    proxy_pool_size: int,
    near_optimal_fraction: float,
) -> None:
    """
    Save only the cumulative mean±std through CompletedRuns.

    There are no Current* columns and no historical per-run columns.  Online
    statistics are maintained in memory by RunningMetricStats.
    """
    nop_percent = 100.0 * float(near_optimal_fraction)
    nop_label = f"NOP@{nop_percent:g}%"

    metadata = [
        f"Dataset: {dataset}",
        f"Tasks: {int(task_count)}",
        f"TotalPipelineRuns: {int(total_runs)}",
        f"CompletedRuns: {int(completed_runs)}",
        f"OptimalCount: {int(optimal_candidate_count)}",
        f"ProxyPoolSize: {int(proxy_pool_size)}",
        f"Delta: {float(dp_para['max_utility_degradation']):.6f}",
        (
            "Phase1: "
            f"{dp_para['pool_construction_mechanism']}"
            f"(eps={float(dp_para['pool_construction_epsilon']):.6f}, "
            f"tau={float(dp_para['pool_construction_clip_bound']):.6f})"
        ),
        "Utility: (Q2+Q3)/2",
        (
            f"{nop_label}: distribution-level probability mass in the "
            "current candidate pool; "
            "e_j=(u_max-u_j)/(u_max-u_min), "
            f"near-optimal iff e_j <= {float(near_optimal_fraction):.6f}; "
            "NOP=100*sum(P_j over near-optimal candidates)"
        ),
        (
            "Metrics: NSD(lower), "
            f"{nop_label}(higher), OPT(higher), QSD(lower), "
            "MeanQD(lower), MaxQD(lower)"
        ),
        (
            "QSD/MeanQD/MaxQD reference: empirical mid-rank in the COMPLETE "
            "public q2/q3 service category; Exact=0"
        ),
        "Storage: cumulative Mean±Std through CompletedRuns only",
        "",
    ]

    columns = [
        "Mechanism",
        "Epsilon",
        "ClipBound",
        "NSD(Mean±Std)",
        f"{nop_label}(Mean±Std)",
        "OPT%(Mean±Std)",
        "QSD(Mean±Std)",
        "MeanQD(Mean±Std)",
        "MaxQD(Mean±Std)",
    ]

    mechanism_order = {"Non_DP": 0, "PNF": 1, "EM": 2}

    def sort_key(setting):
        mechanism, epsilon, clip_bound = setting
        return (
            mechanism_order.get(mechanism, 99),
            -1.0 if epsilon is None else float(epsilon),
            -1.0 if clip_bound is None else float(clip_bound),
        )

    lines = metadata + ["\t".join(columns)]

    for setting in sorted(running_stats, key=sort_key):
        mechanism, epsilon, clip_bound = setting
        stats = running_stats[setting]

        if stats.count != completed_runs:
            raise RuntimeError(
                f"Setting {setting} has statistics for {stats.count} runs, "
                f"expected {completed_runs}."
            )

        nsd_mean, nsd_std = stats.mean_std("normalized_sequence_deviation")
        nop_mean, nop_std = stats.mean_std("near_optimal_probability")
        opt_mean, opt_std = stats.mean_std("optimality_percent")
        qsd_mean, qsd_std = stats.mean_std("qsd")
        mean_qd_mean, mean_qd_std = stats.mean_std("mean_quantile_deviation")
        max_qd_mean, max_qd_std = stats.mean_std("max_quantile_deviation")

        row = [
            mechanism,
            "NA" if epsilon is None else f"{float(epsilon):.6f}",
            "NA" if clip_bound is None else f"{float(clip_bound):.6f}",
            f"{nsd_mean:.10f} ± {nsd_std:.10f}",
            f"{nop_mean:.6f} ± {nop_std:.6f}",
            f"{opt_mean:.6f} ± {opt_std:.6f}",
            f"{qsd_mean:.10f} ± {qsd_std:.10f}",
            f"{mean_qd_mean:.10f} ± {mean_qd_std:.10f}",
            f"{max_qd_mean:.10f} ± {max_qd_std:.10f}",
        ]
        lines.append("\t".join(row))

    _atomic_write_text(output_path, "\n".join(lines) + "\n")

def run_dp_qosc(dataset: str, cfg) -> None:
    validate_dp_para(dp_para)

    reference_path = cfg.exact_reference_path(dataset)
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Formal v5 Exact reference not found:\n"
            f"{reference_path}\n\n"
            "Run Benchmark.py first."
        )

    reference = load_exact_reference(reference_path)
    all_tasks = load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=cfg.test_only,
        min_candidates=1,
    )
    target_tasks, records = verify_reference(
        reference,
        dataset,
        cfg,
        all_tasks,
    )

    if cfg.ga_max_exact_tasks is not None:
        records = records[: cfg.ga_max_exact_tasks]
        selected_ids = {int(record["task_index"]) for record in records}
        target_tasks = [
            task
            for task in target_tasks
            if int(task.task_index) in selected_ids
        ]

    if not records:
        raise RuntimeError("No Exact benchmark task is available.")

    task_map = {
        int(task.task_index): task
        for task in target_tasks
    }
    record_map = {
        int(record["task_index"]): record
        for record in records
    }
    target_task_indices = [
        int(record["task_index"])
        for record in records
    ]

    epsilon_values = [
        float(value)
        for value in dp_para["epsilon"]
    ]
    noise_types = [
        str(value)
        for value in dp_para["noise_types"]
    ]
    clip_bounds = [
        float(value)
        for value in dp_para["clip_bounds"]
    ]
    optimal_candidate_count = int(
        dp_para["optimal_candidate_count"]
    )
    pipeline_runs = int(dp_para["pipeline_runs"])
    dataset_settings = dataset_dp_settings(dataset, dp_para)
    proxy_pool_size = int(dataset_settings["proxy_pool_size"])
    near_optimal_fraction = float(
        dataset_settings["near_optimal_probability_fraction"]
    )

    output_path = dp_result_path(dataset, cfg)
    running_stats: Dict[SettingKey, RunningMetricStats] = {}
    total_start = time.time()

    _, service_path = resolve_dataset_paths(
        dataset,
        cfg.node_file,
        cfg.service_file,
    )
    service_feature = json.loads(
        service_path.read_text(encoding="utf-8")
    )
    public_qos_ranges = build_public_qos_ranges(
        service_feature
    )
    quantile_reference = build_public_quantile_reference(
        service_feature
    )

    print(
        f"Dataset={dataset} | "
        f"Tasks={len(target_task_indices)} | "
        f"PipelineRuns={pipeline_runs} | "
        f"OptimalCount={optimal_candidate_count} | "
        f"ProxyPoolSize={proxy_pool_size} | "
        f"NOP@{100.0 * near_optimal_fraction:g}% | "
        f"Delta={float(dp_para['max_utility_degradation']):.4f} | "
        f"Phase1={dp_para['pool_construction_mechanism']}"
        f"(eps={float(dp_para['pool_construction_epsilon']):.2f}, "
        f"tau={float(dp_para['pool_construction_clip_bound']):.2f}) | "
        "Utility=(Q2+Q3)/2"
    )

    for run_index in range(1, pipeline_runs + 1):
        run_start = time.time()
        run_accumulators: Dict[
            SettingKey,
            MetricAccumulator,
        ] = {}

        for task_index in tqdm(
            target_task_indices,
            desc=f"Pipeline {run_index}/{pipeline_runs}",
        ):
            task = task_map[task_index]
            exact_record = record_map[task_index]

            source_seed = (
                cfg.ga_seed
                + int(run_index) * 10_000_019
                + int(task_index)
            )
            source = build_candidate_source(
                task=task,
                cfg=cfg,
                seed=source_seed,
                optimal_count=optimal_candidate_count,
                proxy_pool_size=proxy_pool_size,
            )
            pool = build_representative_pool(
                source=source,
                optimal_count=optimal_candidate_count,
                para=dp_para,
            )

            nop_grid = near_optimal_probability_grid(
                pool=pool,
                mechanisms=noise_types,
                epsilon_values=epsilon_values,
                clip_bounds=clip_bounds,
                near_optimal_fraction=near_optimal_fraction,
            )

            non_dp_setting: SettingKey = (
                "Non_DP",
                None,
                None,
            )
            non_dp_candidate = select_candidate(
                pool=pool,
                mechanism="Non_DP",
            )
            non_dp_metrics = evaluate_setting_metrics(
                selected=non_dp_candidate,
                task=task,
                exact_record=exact_record,
                service_feature=service_feature,
                public_qos_ranges=public_qos_ranges,
                quantile_reference=quantile_reference,
                near_optimal_probability=100.0,
            )
            run_accumulators.setdefault(
                non_dp_setting,
                MetricAccumulator(),
            ).add(non_dp_metrics)

            for epsilon in epsilon_values:
                for clip_bound in clip_bounds:
                    for mechanism in noise_types:
                        setting: SettingKey = (
                            mechanism,
                            epsilon,
                            clip_bound,
                        )
                        selected = select_candidate(
                            pool=pool,
                            mechanism=mechanism,
                            epsilon=epsilon,
                            clip_bound=clip_bound,
                        )
                        metrics = evaluate_setting_metrics(
                            selected=selected,
                            task=task,
                            exact_record=exact_record,
                            service_feature=service_feature,
                            public_qos_ranges=public_qos_ranges,
                            quantile_reference=quantile_reference,
                            near_optimal_probability=nop_grid[setting],
                        )
                        run_accumulators.setdefault(
                            setting,
                            MetricAccumulator(),
                        ).add(metrics)

        current_run_results = {
            setting: accumulator.mean()
            for setting, accumulator
            in run_accumulators.items()
        }

        for setting, result in current_run_results.items():
            running_stats.setdefault(
                setting,
                RunningMetricStats(),
            ).update(result)

        write_dp_results(
            output_path=output_path,
            dataset=dataset,
            running_stats=running_stats,
            completed_runs=run_index,
            total_runs=pipeline_runs,
            task_count=len(target_task_indices),
            optimal_candidate_count=optimal_candidate_count,
            proxy_pool_size=proxy_pool_size,
            near_optimal_fraction=near_optimal_fraction,
        )

        print(
            f"Run {run_index}/{pipeline_runs} finished | "
            f"NSD/NOP/OPT/QSD/MeanQD/MaxQD saved to {output_path} | "
            f"run_elapsed={time.time() - run_start:.2f}s | "
            f"total_elapsed={time.time() - total_start:.2f}s"
        )

    print(
        f"All DP results saved to: {output_path}"
    )
    print(
        f"Total elapsed: {time.time() - total_start:.2f}s"
    )




def joint_ablation_result_path(dataset: str, cfg) -> Path:
    dataset_safe = (
        str(dataset)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
    )
    result_dir = Path(__file__).resolve().parent / cfg.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{dataset_safe}_delta_optcount_ablation_results.txt"


def write_joint_ablation_results(
    output_path: Path,
    dataset: str,
    running_stats: Dict[
        Tuple[
            float,
            int,
            str,
            Optional[float],
            Optional[float],
        ],
        RunningMetricStats,
    ],
    representative_stats: Dict[
        Tuple[float, int],
        RunningScalarStats,
    ],
    completed_runs: int,
    total_runs: int,
    task_count: int,
    proxy_pool_size: int,
    near_optimal_fraction: float,
    epsilon_values: List[float],
    clip_bounds: List[float],
) -> None:
    """
    Save the COMPLETE Delta x OptCount x epsilon x tau x mechanism grid.

    Only cumulative mean±std through CompletedRuns is stored.  Neither Delta nor
    OptCount is fixed while sweeping the other.
    """
    nop_percent = 100.0 * float(near_optimal_fraction)
    nop_label = f"NOP@{nop_percent:g}%"
    delta_values = [
        float(value)
        for value in dp_para["delta_ablation_values"]
    ]
    optcount_values = [
        int(value)
        for value in dp_para["optimal_count_ablation_values"]
    ]

    metadata = [
        f"Dataset: {dataset}",
        f"Tasks: {int(task_count)}",
        f"TotalPipelineRuns: {int(total_runs)}",
        f"CompletedRuns: {int(completed_runs)}",
        "DeltaValues: " + ",".join(f"{value:g}" for value in delta_values),
        "OptCountValues: " + ",".join(str(value) for value in optcount_values),
        f"ProxyPoolSize: {int(proxy_pool_size)}",
        "EpsilonValues: " + ",".join(f"{value:g}" for value in epsilon_values),
        "ClipBounds: " + ",".join(f"{value:g}" for value in clip_bounds),
        (
            "Phase1Reference: Q(Psi_opt) under "
            f"{dp_para['pool_construction_mechanism']}"
            f"(eps={float(dp_para['pool_construction_epsilon']):.6f}, "
            f"tau={float(dp_para['pool_construction_clip_bound']):.6f})"
        ),
        "DeltaConstraint: Q(Psi) >= (1-delta) * Q(Psi_opt)",
        "Utility: (Q2+Q3)/2",
        (
            f"{nop_label}: distribution-level probability mass in the "
            "current candidate pool; "
            "e_j=(u_max-u_j)/(u_max-u_min), "
            f"near-optimal iff e_j <= {float(near_optimal_fraction):.6f}; "
            "NOP=100*sum(P_j over near-optimal candidates)"
        ),
        (
            "Metrics: NSD(lower), "
            f"{nop_label}(higher), OPT(higher), QSD(lower), "
            "MeanQD(lower), MaxQD(lower)"
        ),
        (
            "RepresentativeCount: mean number of Psi_rep sequences per task; "
            "reported as pipeline-run Mean±Std for every (Delta,OptCount)"
        ),
        (
            "JointAblation: every Delta x OptCount combination is evaluated; "
            "no fixed counterpart value is used"
        ),
        (
            "CandidateSourceReuse: within one task/run, a source built with "
            "max(OptCount) is reused for all Delta x OptCount combinations"
        ),
        "Storage: cumulative Mean±Std through CompletedRuns only",
        "",
    ]

    representative_lines = [
        "RepresentativeCountByDeltaOptCount",
        "Delta\tOptCount\tRepresentativeCount(Mean±Std)",
    ]
    for delta in delta_values:
        for opt_count in optcount_values:
            key = (float(delta), int(opt_count))
            stats = representative_stats[key]
            if stats.count != completed_runs:
                raise RuntimeError(
                    f"Representative-count setting {key} has statistics for "
                    f"{stats.count} runs, expected {completed_runs}."
                )
            rep_mean, rep_std = stats.mean_std()
            representative_lines.append(
                f"{float(delta):.6f}\t{int(opt_count)}\t"
                f"{rep_mean:.6f} ± {rep_std:.6f}"
            )

    columns = [
        "Delta",
        "OptCount",
        "Mechanism",
        "Epsilon",
        "ClipBound",
        "NSD(Mean±Std)",
        f"{nop_label}(Mean±Std)",
        "OPT%(Mean±Std)",
        "QSD(Mean±Std)",
        "MeanQD(Mean±Std)",
        "MaxQD(Mean±Std)",
    ]

    mechanism_order = {"Non_DP": 0, "PNF": 1, "EM": 2}

    def sort_key(setting):
        delta, opt_count, mechanism, epsilon, clip_bound = setting
        return (
            float(delta),
            int(opt_count),
            mechanism_order.get(mechanism, 99),
            -1.0 if epsilon is None else float(epsilon),
            -1.0 if clip_bound is None else float(clip_bound),
        )

    lines = metadata + representative_lines + ["", "\t".join(columns)]

    for setting in sorted(running_stats, key=sort_key):
        delta, opt_count, mechanism, epsilon, clip_bound = setting
        stats = running_stats[setting]
        if stats.count != completed_runs:
            raise RuntimeError(
                f"Ablation setting {setting} has statistics for {stats.count} runs, "
                f"expected {completed_runs}."
            )

        nsd_mean, nsd_std = stats.mean_std("normalized_sequence_deviation")
        nop_mean, nop_std = stats.mean_std("near_optimal_probability")
        opt_mean, opt_std = stats.mean_std("optimality_percent")
        qsd_mean, qsd_std = stats.mean_std("qsd")
        mean_qd_mean, mean_qd_std = stats.mean_std("mean_quantile_deviation")
        max_qd_mean, max_qd_std = stats.mean_std("max_quantile_deviation")

        row = [
            f"{float(delta):.6f}",
            str(int(opt_count)),
            mechanism,
            "NA" if epsilon is None else f"{float(epsilon):.6f}",
            "NA" if clip_bound is None else f"{float(clip_bound):.6f}",
            f"{nsd_mean:.10f} ± {nsd_std:.10f}",
            f"{nop_mean:.6f} ± {nop_std:.6f}",
            f"{opt_mean:.6f} ± {opt_std:.6f}",
            f"{qsd_mean:.10f} ± {qsd_std:.10f}",
            f"{mean_qd_mean:.10f} ± {mean_qd_std:.10f}",
            f"{max_qd_mean:.10f} ± {max_qd_std:.10f}",
        ]
        lines.append("\t".join(row))

    _atomic_write_text(output_path, "\n".join(lines) + "\n")


def run_joint_ablation(dataset: str, cfg) -> None:
    """
    Jointly evaluate the COMPLETE:

        Delta x OptCount x epsilon x tau x mechanism

    grid.  This intentionally avoids fixing OptCount during a delta sweep or
    fixing delta during an OptCount sweep.  The user can select/report the
    desired slice after all results are available.
    """
    validate_dp_para(dp_para)
    dataset_settings = dataset_dp_settings(dataset, dp_para)
    proxy_pool_size = int(dataset_settings["proxy_pool_size"])
    near_optimal_fraction = float(
        dataset_settings["near_optimal_probability_fraction"]
    )

    delta_values = [
        float(value)
        for value in dp_para["delta_ablation_values"]
    ]
    optcount_values = [
        int(value)
        for value in dp_para["optimal_count_ablation_values"]
    ]
    epsilon_values = [float(value) for value in dp_para["epsilon"]]
    clip_bounds = [float(value) for value in dp_para["clip_bounds"]]
    mechanisms = [str(value) for value in dp_para["noise_types"]]
    pipeline_runs = int(dp_para["pipeline_runs"])
    max_optcount = max(optcount_values)

    reference_path = cfg.exact_reference_path(dataset)
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Formal v5 Exact reference not found:\n"
            f"{reference_path}\n\nRun Benchmark.py first."
        )

    reference = load_exact_reference(reference_path)
    all_tasks = load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=cfg.test_only,
        min_candidates=1,
    )
    target_tasks, records = verify_reference(
        reference,
        dataset,
        cfg,
        all_tasks,
    )

    if cfg.ga_max_exact_tasks is not None:
        records = records[: cfg.ga_max_exact_tasks]
        selected_ids = {int(record["task_index"]) for record in records}
        target_tasks = [
            task
            for task in target_tasks
            if int(task.task_index) in selected_ids
        ]

    if not records:
        raise RuntimeError("No Exact benchmark task is available.")

    task_map = {int(task.task_index): task for task in target_tasks}
    record_map = {int(record["task_index"]): record for record in records}
    target_task_indices = [int(record["task_index"]) for record in records]

    _, service_path = resolve_dataset_paths(
        dataset,
        cfg.node_file,
        cfg.service_file,
    )
    service_feature = json.loads(service_path.read_text(encoding="utf-8"))
    public_qos_ranges = build_public_qos_ranges(service_feature)
    quantile_reference = build_public_quantile_reference(service_feature)

    output_path = joint_ablation_result_path(dataset, cfg)

    running_stats: Dict[
        Tuple[
            float,
            int,
            str,
            Optional[float],
            Optional[float],
        ],
        RunningMetricStats,
    ] = {}

    representative_stats: Dict[
        Tuple[float, int],
        RunningScalarStats,
    ] = {
        (float(delta), int(opt_count)): RunningScalarStats()
        for delta in delta_values
        for opt_count in optcount_values
    }

    total_start = time.time()
    settings_per_pool = (
        1 + len(mechanisms) * len(epsilon_values) * len(clip_bounds)
    )
    total_pool_settings = len(delta_values) * len(optcount_values)

    print(
        f"Joint Delta x OptCount ablation | Dataset={dataset} | "
        f"Tasks={len(target_task_indices)} | Runs={pipeline_runs} | "
        f"ProxyPoolSize={proxy_pool_size} | "
        f"NOP@{100.0 * near_optimal_fraction:g}% | "
        f"Deltas={delta_values} | OptCounts={optcount_values} | "
        f"Eps={epsilon_values} | Tau={clip_bounds} | "
        f"Mechanisms={mechanisms} | "
        f"DeltaOptCountPairs={total_pool_settings} | "
        f"SettingsPerPair={settings_per_pool}"
    )

    for run_index in range(1, pipeline_runs + 1):
        run_start = time.time()
        run_accumulators: Dict[
            Tuple[
                float,
                int,
                str,
                Optional[float],
                Optional[float],
            ],
            MetricAccumulator,
        ] = {}
        run_representative_counts: Dict[
            Tuple[float, int],
            ScalarAccumulator,
        ] = {
            (float(delta), int(opt_count)): ScalarAccumulator()
            for delta in delta_values
            for opt_count in optcount_values
        }

        for task_index in tqdm(
            target_task_indices,
            desc=f"Joint ablation {run_index}/{pipeline_runs}",
        ):
            task = task_map[task_index]
            exact_record = record_map[task_index]
            source_seed = (
                cfg.ga_seed
                + int(run_index) * 10_000_019
                + int(task_index)
            )

            source = build_candidate_source(
                task=task,
                cfg=cfg,
                seed=source_seed,
                optimal_count=max_optcount,
                proxy_pool_size=proxy_pool_size,
            )

            for opt_count in optcount_values:
                for delta in delta_values:
                    para_local = dict(dp_para)
                    para_local["max_utility_degradation"] = float(delta)

                    pool = build_representative_pool(
                        source=source,
                        optimal_count=int(opt_count),
                        para=para_local,
                    )

                    nop_grid = near_optimal_probability_grid(
                        pool=pool,
                        mechanisms=mechanisms,
                        epsilon_values=epsilon_values,
                        clip_bounds=clip_bounds,
                        near_optimal_fraction=near_optimal_fraction,
                    )

                    pair_key = (float(delta), int(opt_count))
                    run_representative_counts[pair_key].add(
                        float(pool.representative_count)
                    )

                    non_dp_candidate = select_candidate(
                        pool=pool,
                        mechanism="Non_DP",
                    )
                    non_dp_metrics = evaluate_setting_metrics(
                        selected=non_dp_candidate,
                        task=task,
                        exact_record=exact_record,
                        service_feature=service_feature,
                        public_qos_ranges=public_qos_ranges,
                        quantile_reference=quantile_reference,
                        near_optimal_probability=100.0,
                    )
                    non_dp_setting = (
                        float(delta),
                        int(opt_count),
                        "Non_DP",
                        None,
                        None,
                    )
                    run_accumulators.setdefault(
                        non_dp_setting,
                        MetricAccumulator(),
                    ).add(non_dp_metrics)

                    for epsilon in epsilon_values:
                        for clip_bound in clip_bounds:
                            for mechanism in mechanisms:
                                metric_setting: SettingKey = (
                                    mechanism,
                                    epsilon,
                                    clip_bound,
                                )
                                selected = select_candidate(
                                    pool=pool,
                                    mechanism=mechanism,
                                    epsilon=epsilon,
                                    clip_bound=clip_bound,
                                )
                                metrics = evaluate_setting_metrics(
                                    selected=selected,
                                    task=task,
                                    exact_record=exact_record,
                                    service_feature=service_feature,
                                    public_qos_ranges=public_qos_ranges,
                                    quantile_reference=quantile_reference,
                                    near_optimal_probability=nop_grid[metric_setting],
                                )
                                ablation_setting = (
                                    float(delta),
                                    int(opt_count),
                                    mechanism,
                                    float(epsilon),
                                    float(clip_bound),
                                )
                                run_accumulators.setdefault(
                                    ablation_setting,
                                    MetricAccumulator(),
                                ).add(metrics)

        current_run_results = {
            setting: accumulator.mean()
            for setting, accumulator in run_accumulators.items()
        }
        for setting, result in current_run_results.items():
            running_stats.setdefault(
                setting,
                RunningMetricStats(),
            ).update(result)

        for pair_key, accumulator in run_representative_counts.items():
            representative_stats[pair_key].update(accumulator.mean())

        write_joint_ablation_results(
            output_path=output_path,
            dataset=dataset,
            running_stats=running_stats,
            representative_stats=representative_stats,
            completed_runs=run_index,
            total_runs=pipeline_runs,
            task_count=len(target_task_indices),
            proxy_pool_size=proxy_pool_size,
            near_optimal_fraction=near_optimal_fraction,
            epsilon_values=epsilon_values,
            clip_bounds=clip_bounds,
        )

        print(
            f"Joint ablation run {run_index}/{pipeline_runs} finished | "
            f"saved to {output_path} | "
            f"run_elapsed={time.time() - run_start:.2f}s | "
            f"total_elapsed={time.time() - total_start:.2f}s"
        )

    print(f"All joint ablation results saved to: {output_path}")
    print(f"Total elapsed: {time.time() - total_start:.2f}s")

def run_exact_quality(dataset: str, cfg) -> None:
    reference_path = cfg.exact_reference_path(dataset)

    if not reference_path.is_file():
        raise FileNotFoundError(
            "Formal v5 Exact reference not found:\n"
            f"{reference_path}\n\n"
            "Run Benchmark.py first."
        )

    reference = load_exact_reference(reference_path)
    all_tasks = load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=cfg.test_only,
        min_candidates=1,
    )
    target_tasks, records = verify_reference(
        reference,
        dataset,
        cfg,
        all_tasks,
    )
    task_map = {
        int(task.task_index): task
        for task in target_tasks
    }

    if cfg.ga_max_exact_tasks is not None:
        records = records[: cfg.ga_max_exact_tasks]

    exact_distances = []
    ga_distances = []
    ga_optimalities = []
    ga_gaps = []

    for record in tqdm(
        records,
        desc="SLA-GA vs Exact v5",
    ):
        task_index = int(record["task_index"])
        task = task_map[task_index]
        exact_distance = float(record["exact_distance"])

        model = build_ga(
            task,
            cfg,
            cfg.ga_seed + task_index,
        )
        _, ga_evaluation = model.search()

        if ga_evaluation.violations != 0:
            continue

        exact_distances.append(exact_distance)
        ga_distances.append(ga_evaluation.distance)
        ga_optimalities.append(
            boundary_optimality(
                exact_distance,
                ga_evaluation.distance,
            )
        )
        ga_gaps.append(
            distance_gap(
                exact_distance,
                ga_evaluation.distance,
            )
        )

    print(f"Dataset: {dataset}")
    print(f"Tasks: {len(records)}")
    print(f"Mean Exact D*: {statistics.mean(exact_distances):.10f}")
    print(f"Mean GA D: {statistics.mean(ga_distances):.10f}")
    print(
        "Mean Boundary Optimality: "
        f"{statistics.mean(ga_optimalities):.10f}"
    )
    print(f"Mean Distance Gap: {statistics.mean(ga_gaps):.10f}")


def run_full_scale(dataset: str, cfg) -> None:
    tasks = load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=cfg.test_only,
        min_candidates=1,
    )

    distances = []
    performances = []

    for task in tqdm(tasks, desc="Full-scale SLA-GA"):
        model = build_ga(
            task,
            cfg,
            cfg.ga_seed + task.task_index,
        )
        _, evaluation = model.search()
        distances.append(evaluation.distance)
        performances.append(evaluation.performance_objective)

    print(f"Dataset: {dataset}")
    print(f"Tasks: {len(tasks)}")
    print(f"Mean D: {statistics.mean(distances):.10f}")
    print(f"Mean F: {statistics.mean(performances):.10f}")



def run(dataset: str, mode: str = "main") -> None:
    cfg = load_config(dataset=dataset)

    if dp_para.get("use_dp", False):
        if mode == "ablation":
            run_joint_ablation(dataset, cfg)
        else:
            run_dp_qosc(dataset, cfg)
        return

    if mode != "main":
        raise ValueError("DP ablation mode requires dp_para['use_dp']=True.")

    if cfg.ga_mode == "exact_quality":
        run_exact_quality(dataset, cfg)
    elif cfg.ga_mode == "full_scale":
        run_full_scale(dataset, cfg)
    else:
        raise ValueError(cfg.ga_mode)

if __name__ == "__main__":
    _dataset, _mode = parse_arguments()
    run(_dataset, _mode)
