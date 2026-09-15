from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from config_loader import load_config
from dp_perturb import dp_para, validate_dp_para
from dp_qosc import (
    SequenceCandidate,
    build_candidate_source,
    build_representative_pool,
    select_candidate,
)
from loadData_GA import SLATask, Service, load_sla_tasks, resolve_dataset_paths
from sla_core import SLAOnDemandGA, file_sha256


CACHE_VERSION = 2
DIST_TOL = 1e-12
CACHE_SAVE_EVERY_TASKS = 50
DEFAULT_DELTA_VALUES = (0.02, 0.04, 0.06, 0.08, 0.10)
SUPPORTED_DISTANCES = ("hamming", "jaccard", "qos")

ServiceIdentity = Tuple[int, int]
AlignedSettingKey = Tuple[
    str,                    # pool variant: OPT_ONLY / REPRESENTATIVE
    Optional[float],        # delta, NA for OPT_ONLY
    str,                    # distance
    str,                    # release method
    Optional[float],        # epsilon
    Optional[float],        # clip bound
]
MatchedSettingKey = Tuple[
    float,                  # representative-pool delta whose K is used
    str,                    # distance
    str,                    # release method of OPT_ONLY release
    Optional[float],        # epsilon
    Optional[float],        # clip bound
]

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

POOL_METRIC_KEYS = (
    "pool_size",
    "representative_count",
    "representativeness",
    "phase1_expected_utility",
    "phase1_reference_utility",
    "phase1_actual_utility_degradation_percent",
)


# =============================================================================
# Data structures
# =============================================================================
@dataclass(frozen=True)
class LocalRequirement:
    category_id: int
    q2_lower: float
    q2_upper: float
    q3_lower: float
    q3_upper: float


@dataclass(frozen=True)
class RequirementHypothesis:
    positions: Tuple[LocalRequirement, ...]


@dataclass(frozen=True)
class AttackTemplate:
    hypothesis: RequirementHypothesis
    service_ids: Tuple[ServiceIdentity, ...]
    q2_product: float
    q3_product: float


@dataclass(frozen=True)
class ReleaseObservation:
    service_ids: Tuple[ServiceIdentity, ...]
    q2_product: float
    q3_product: float


@dataclass(frozen=True)
class InferenceResult:
    winner_indices: Tuple[int, ...]
    distances: Tuple[float, ...]


@dataclass
class TrainingRequirementBank:
    samples_by_category: Dict[int, Tuple[LocalRequirement, ...]]

    unique_by_category: Dict[int, Tuple[LocalRequirement, ...]]


@dataclass
class ExpandableAttackModel:
    task_index: int
    true_hypothesis: RequirementHypothesis
    templates: List[AttackTemplate]
    next_profile_rank: int

    def true_template(self) -> AttackTemplate:
        matches = [
            template
            for template in self.templates
            if same_hypothesis(template.hypothesis, self.true_hypothesis)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Task {self.task_index}: expected exactly one true hypothesis, "
                f"found {len(matches)}."
            )
        return matches[0]


@dataclass
class PoolView:
    candidates: List[SequenceCandidate]


@dataclass
class MetricAccumulator:
    keys: Tuple[str, ...]
    sums: Dict[str, float] = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.sums = {key: 0.0 for key in self.keys}

    def add(self, metrics: Dict[str, float]) -> None:
        for key in self.keys:
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"Metric {key} is not finite: {value}")
            self.sums[key] += value
        self.count += 1

    def mean(self) -> Dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("Cannot average an empty metric accumulator.")
        return {key: self.sums[key] / self.count for key in self.keys}


@dataclass
class RunningMetricStats:
    keys: Tuple[str, ...]
    count: int = 0
    means: Dict[str, float] = field(init=False)
    m2: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.means = {key: 0.0 for key in self.keys}
        self.m2 = {key: 0.0 for key in self.keys}

    def update(self, metrics: Dict[str, float]) -> None:
        new_count = self.count + 1
        for key in self.keys:
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

def normalize_interval(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"QoS interval must contain two values: {values!r}")

    lower = float(values[0])
    upper = float(values[1])
    if lower > upper:
        lower, upper = upper, lower

    if not (0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0):
        raise ValueError(f"QoS interval outside [0,1]: {(lower, upper)}")
    if lower >= upper:
        raise ValueError(f"QoS interval requires lower < upper: {(lower, upper)}")

    return lower, upper


def requirement_key(
    requirement: LocalRequirement,
) -> Tuple[int, float, float, float, float]:
    return (
        int(requirement.category_id),
        float(requirement.q2_lower),
        float(requirement.q2_upper),
        float(requirement.q3_lower),
        float(requirement.q3_upper),
    )


def hypothesis_key(
    hypothesis: RequirementHypothesis,
) -> Tuple[Tuple[int, float, float, float, float], ...]:
    return tuple(requirement_key(req) for req in hypothesis.positions)


def same_hypothesis(
    left: RequirementHypothesis,
    right: RequirementHypothesis,
    tolerance: float = DIST_TOL,
) -> bool:
    if len(left.positions) != len(right.positions):
        return False

    for lhs, rhs in zip(left.positions, right.positions):
        if int(lhs.category_id) != int(rhs.category_id):
            return False
        for a, b in (
            (lhs.q2_lower, rhs.q2_lower),
            (lhs.q2_upper, rhs.q2_upper),
            (lhs.q3_lower, rhs.q3_lower),
            (lhs.q3_upper, rhs.q3_upper),
        ):
            if abs(float(a) - float(b)) > tolerance:
                return False
    return True


def local_requirement_distance(
    left: LocalRequirement,
    right: LocalRequirement,
) -> float:

    if int(left.category_id) != int(right.category_id):
        raise ValueError("Local requirement distance requires the same category.")

    return 0.25 * (
        abs(float(left.q2_lower) - float(right.q2_lower))
        + abs(float(left.q2_upper) - float(right.q2_upper))
        + abs(float(left.q3_lower) - float(right.q3_lower))
        + abs(float(left.q3_upper) - float(right.q3_upper))
    )


def requirement_profile_nmae(
    estimate: RequirementHypothesis,
    truth: RequirementHypothesis,
) -> float:

    if len(estimate.positions) != len(truth.positions):
        raise ValueError("Requirement profiles have different lengths.")

    absolute_error = 0.0
    endpoint_count = 0
    for estimated, actual in zip(estimate.positions, truth.positions):
        if int(estimated.category_id) != int(actual.category_id):
            raise RuntimeError(
                "Cannot compare requirement profiles with different workflows."
            )
        absolute_error += (
            abs(float(estimated.q2_lower) - float(actual.q2_lower))
            + abs(float(estimated.q2_upper) - float(actual.q2_upper))
            + abs(float(estimated.q3_lower) - float(actual.q3_lower))
            + abs(float(estimated.q3_upper) - float(actual.q3_upper))
        )
        endpoint_count += 4

    if endpoint_count <= 0:
        raise RuntimeError("Cannot compute NMAE for an empty profile.")
    return float(absolute_error / endpoint_count)


# =============================================================================
# Public repository and training background
# =============================================================================
def load_public_service_repository(
    dataset: str,
    node_filename: str,
    service_filename: str,
) -> Dict[int, Tuple[Service, ...]]:
    _, service_path = resolve_dataset_paths(
        dataset,
        node_filename,
        service_filename,
    )

    raw_repository = json.loads(service_path.read_text(encoding="utf-8"))
    repository: Dict[int, Tuple[Service, ...]] = {}

    for raw_category_id, raw_services in raw_repository.items():
        category_id = int(raw_category_id)
        services: List[Service] = []
        for concrete_index, raw_service in enumerate(raw_services):
            if len(raw_service) < 4:
                raise ValueError(
                    f"Category {category_id}, service {concrete_index}: fewer "
                    "than four QoS values."
                )
            services.append(
                (
                    float(raw_service[-4]),
                    float(raw_service[-3]),
                    float(raw_service[-2]),
                    float(raw_service[-1]),
                    category_id,
                    int(concrete_index),
                )
            )
        if not services:
            raise RuntimeError(f"Public category {category_id} is empty.")
        repository[category_id] = tuple(services)

    if not repository:
        raise RuntimeError("Public service repository is empty.")
    return repository


def filter_public_services(
    repository: Dict[int, Tuple[Service, ...]],
    requirement: LocalRequirement,
) -> List[Service]:
    category_id = int(requirement.category_id)
    if category_id not in repository:
        raise KeyError(f"Public category {category_id} is absent.")

    return [
        service
        for service in repository[category_id]
        if (
            float(requirement.q2_lower)
            <= float(service[2])
            <= float(requirement.q2_upper)
            and float(requirement.q3_lower)
            <= float(service[3])
            <= float(requirement.q3_upper)
        )
    ]


def load_training_requirement_bank(
    dataset: str,
    node_filename: str,
    service_filename: str,
    repository: Dict[int, Tuple[Service, ...]],
) -> TrainingRequirementBank:

    node_path, _ = resolve_dataset_paths(
        dataset,
        node_filename,
        service_filename,
    )
    nodefeatures = json.loads(node_path.read_text(encoding="utf-8"))
    split_index = len(nodefeatures) * 3 // 4
    training_tasks = nodefeatures[:split_index]

    samples: Dict[int, List[LocalRequirement]] = {}
    unique: Dict[
        int,
        Dict[Tuple[int, float, float, float, float], LocalRequirement],
    ] = {}

    for nodes in training_tasks:
        for node in nodes:
            if len(node) < 7:
                raise ValueError(
                    "Malformed node while building training requirement bank."
                )

            prefix = list(node[:-6])
            if 1 not in prefix:
                raise ValueError("Training node has no active one-hot entry.")
            category_id = int(prefix.index(1))

            if category_id == 0:
                continue

            q2_lower, q2_upper = normalize_interval(node[-5:-3])
            q3_lower, q3_upper = normalize_interval(node[-2:])
            requirement = LocalRequirement(
                category_id=category_id,
                q2_lower=q2_lower,
                q2_upper=q2_upper,
                q3_lower=q3_lower,
                q3_upper=q3_upper,
            )

            # A plausible hypothesis must define a nonempty local public set.
            if not filter_public_services(repository, requirement):
                continue

            samples.setdefault(category_id, []).append(requirement)
            unique.setdefault(category_id, {})[
                requirement_key(requirement)
            ] = requirement

    samples_by_category = {
        int(category_id): tuple(values)
        for category_id, values in samples.items()
        if values
    }
    unique_by_category = {
        int(category_id): tuple(sorted(values.values(), key=requirement_key))
        for category_id, values in unique.items()
        if values
    }

    if not samples_by_category:
        raise RuntimeError("No empirical training local requirements are available.")

    return TrainingRequirementBank(
        samples_by_category=samples_by_category,
        unique_by_category=unique_by_category,
    )


# =============================================================================
# Robust hypothetical-task reconstruction
# =============================================================================
def rebuild_local_constraint(
    base_constraint,
    requirement: LocalRequirement,
    raw_count: int,
    feasible_count: int,
):

    if not is_dataclass(base_constraint):
        return base_constraint

    fields = getattr(base_constraint, "__dataclass_fields__", {})
    updates = {}
    if "q2" in fields:
        updates["q2"] = (
            float(requirement.q2_lower),
            float(requirement.q2_upper),
        )
    if "q3" in fields:
        updates["q3"] = (
            float(requirement.q3_lower),
            float(requirement.q3_upper),
        )
    if "raw_candidate_count" in fields:
        updates["raw_candidate_count"] = int(raw_count)
    if "feasible_candidate_count" in fields:
        updates["feasible_candidate_count"] = int(feasible_count)
    if "search_candidate_count" in fields:
        updates["search_candidate_count"] = int(feasible_count)
    if "used_random_fallback" in fields:
        updates["used_random_fallback"] = False

    return replace(base_constraint, **updates)


def make_hypothetical_task(
    base_task: SLATask,
    hypothesis: RequirementHypothesis,
    repository: Dict[int, Tuple[Service, ...]],
) -> SLATask:

    if len(hypothesis.positions) != len(base_task.local_constraints):
        raise ValueError(
            f"Task {base_task.task_index}: hypothesis length mismatch."
        )

    services: List[List[Service]] = []
    local_constraints = []

    for position, (base_constraint, requirement) in enumerate(
        zip(base_task.local_constraints, hypothesis.positions)
    ):
        expected_category = int(base_constraint.category_id)
        if int(requirement.category_id) != expected_category:
            raise RuntimeError(
                f"Task {base_task.task_index}, position {position}: hypothesis "
                f"category {requirement.category_id} != public workflow category "
                f"{expected_category}."
            )

        candidates = filter_public_services(repository, requirement)
        if not candidates:
            raise RuntimeError(
                f"Task {base_task.task_index}, position {position}: hypothesis "
                "creates an empty local candidate set."
            )

        raw_count = len(repository[expected_category])
        services.append(candidates)
        local_constraints.append(
            rebuild_local_constraint(
                base_constraint=base_constraint,
                requirement=requirement,
                raw_count=raw_count,
                feasible_count=len(candidates),
            )
        )

    return SLATask(
        task_index=int(base_task.task_index),
        services=services,
        global_q2=base_task.global_q2,
        global_q3=base_task.global_q3,
        local_constraints=local_constraints,
    )


# =============================================================================
# Public conventional predictor CS*(H)
# =============================================================================
def public_attack_seed(task_index: int, ga_seed: int) -> int:
    """Every hypothesis for one workflow uses the SAME public GA seed."""
    return int(ga_seed) + int(task_index) * 10_000_019


def build_public_ga(task: SLATask, cfg, seed: int) -> SLAOnDemandGA:
    return SLAOnDemandGA(
        task,
        population_size=cfg.ga_population,
        max_generations=cfg.ga_max_generations,
        stagnation_patience=cfg.ga_stagnation_patience,
        crossover_rate=cfg.ga_crossover_rate,
        mutation_probability=cfg.ga_mutation_probability,
        elite_count=cfg.ga_elite_count,
        seed=int(seed),
    )


def service_identities(
    sequence: Sequence[Service],
) -> Tuple[ServiceIdentity, ...]:
    return tuple(
        (int(service[4]), int(service[5]))
        for service in sequence
    )


def build_attack_template(
    base_task: SLATask,
    hypothesis: RequirementHypothesis,
    repository: Dict[int, Tuple[Service, ...]],
    cfg,
) -> Optional[AttackTemplate]:
    hypothetical_task = make_hypothetical_task(
        base_task=base_task,
        hypothesis=hypothesis,
        repository=repository,
    )
    model = build_public_ga(
        task=hypothetical_task,
        cfg=cfg,
        seed=public_attack_seed(
            task_index=int(base_task.task_index),
            ga_seed=int(cfg.ga_seed),
        ),
    )
    sequence, evaluation = model.search()
    if evaluation.violations != 0:
        return None

    return AttackTemplate(
        hypothesis=hypothesis,
        service_ids=service_identities(sequence),
        q2_product=float(evaluation.q2_product),
        q3_product=float(evaluation.q3_product),
    )


# =============================================================================
# Public empirical hypothesis stream
# =============================================================================
def position_neighbor_lists(
    base_task: SLATask,
    training_bank: TrainingRequirementBank,
    cfg,
) -> Tuple[Tuple[LocalRequirement, ...], ...]:

    rng = random.Random(
        int(cfg.ga_seed)
        + int(base_task.task_index) * 1_000_003
        + 0xC0A017
    )

    ordered_by_position: List[Tuple[LocalRequirement, ...]] = []
    for position, constraint in enumerate(base_task.local_constraints):
        category_id = int(constraint.category_id)
        samples = training_bank.samples_by_category.get(category_id, ())
        unique = training_bank.unique_by_category.get(category_id, ())
        if not samples or not unique:
            raise RuntimeError(
                f"Task {base_task.task_index}, position {position}: no training "
                f"requirement background for category {category_id}."
            )

        anchor = samples[rng.randrange(len(samples))]
        ordered = sorted(
            unique,
            key=lambda requirement: (
                local_requirement_distance(anchor, requirement),
                requirement_key(requirement),
            ),
        )
        ordered_by_position.append(tuple(ordered))

    return tuple(ordered_by_position)


def profile_from_neighbor_rank(
    ordered_by_position: Sequence[Sequence[LocalRequirement]],
    rank: int,
) -> RequirementHypothesis:
    rank = int(rank)
    positions = []
    for ordered in ordered_by_position:
        if not 0 <= rank < len(ordered):
            raise IndexError(f"Neighbor rank {rank} is unavailable.")
        positions.append(ordered[rank])
    return RequirementHypothesis(positions=tuple(positions))


def build_initial_attack_model(
    base_task: SLATask,
    training_bank: TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    base_count: int,
    cfg,
) -> ExpandableAttackModel:

    base_count = int(base_count)
    if base_count < 2:
        raise ValueError("base_count must be >= 2.")

    ordered_by_position = position_neighbor_lists(
        base_task=base_task,
        training_bank=training_bank,
        cfg=cfg,
    )
    max_rank = min(len(ordered) for ordered in ordered_by_position)
    if max_rank < base_count:
        raise RuntimeError(
            f"Task {base_task.task_index}: at least one category has only "
            f"{max_rank} distinct training requirements, fewer than base_count="
            f"{base_count}."
        )

    accepted: List[AttackTemplate] = []
    seen_profiles = set()
    next_rank = 0

    for rank in range(max_rank):
        next_rank = rank + 1
        hypothesis = profile_from_neighbor_rank(ordered_by_position, rank)
        key = hypothesis_key(hypothesis)
        if key in seen_profiles:
            continue

        template = build_attack_template(
            base_task=base_task,
            hypothesis=hypothesis,
            repository=repository,
            cfg=cfg,
        )
        if template is None:
            continue

        seen_profiles.add(key)
        accepted.append(template)
        if len(accepted) >= base_count:
            break

    if len(accepted) != base_count:
        raise RuntimeError(
            f"Task {base_task.task_index}: built only {len(accepted)}/{base_count} "
            f"feasible public hypotheses from {max_rank} empirical ranks."
        )

    victim_rng = random.Random(
        int(cfg.ga_seed)
        + int(base_task.task_index) * 2_000_003
        + 0xE7A1
    )
    victim_index = victim_rng.randrange(base_count)
    true_hypothesis = accepted[victim_index].hypothesis

    victim_rng.shuffle(accepted)

    return ExpandableAttackModel(
        task_index=int(base_task.task_index),
        true_hypothesis=true_hypothesis,
        templates=accepted,
        next_profile_rank=int(next_rank),
    )


def ensure_attack_template_count(
    attack_model: ExpandableAttackModel,
    target_count: int,
    base_task: SLATask,
    training_bank: TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    cfg,
) -> int:

    target_count = int(target_count)
    if target_count < 2:
        raise ValueError("target_count must be >= 2.")
    if len(attack_model.templates) >= target_count:
        return 0

    ordered_by_position = position_neighbor_lists(
        base_task=base_task,
        training_bank=training_bank,
        cfg=cfg,
    )
    max_rank = min(len(ordered) for ordered in ordered_by_position)

    seen_profiles = {
        hypothesis_key(template.hypothesis)
        for template in attack_model.templates
    }
    start_count = len(attack_model.templates)

    rank = int(attack_model.next_profile_rank)
    while rank < max_rank and len(attack_model.templates) < target_count:
        hypothesis = profile_from_neighbor_rank(ordered_by_position, rank)
        rank += 1
        key = hypothesis_key(hypothesis)
        if key in seen_profiles:
            continue

        template = build_attack_template(
            base_task=base_task,
            hypothesis=hypothesis,
            repository=repository,
            cfg=cfg,
        )
        if template is None:
            continue

        seen_profiles.add(key)
        attack_model.templates.append(template)

    attack_model.next_profile_rank = int(rank)

    if len(attack_model.templates) < target_count:
        raise RuntimeError(
            f"Task {base_task.task_index}: public hypothesis stream exhausted at "
            f"{len(attack_model.templates)} templates, but K={target_count} was "
            "required by the pool-size-aligned evaluation."
        )

    return len(attack_model.templates) - start_count

_WORKER_TASKS: Dict[int, SLATask] = {}
_WORKER_BANK: Optional[TrainingRequirementBank] = None
_WORKER_REPOSITORY: Dict[int, Tuple[Service, ...]] = {}
_WORKER_CFG = None
_WORKER_BASE_COUNT = 0


def init_worker(
    tasks: Sequence[SLATask],
    training_bank: TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    cfg,
    base_count: int,
) -> None:
    global _WORKER_TASKS
    global _WORKER_BANK
    global _WORKER_REPOSITORY
    global _WORKER_CFG
    global _WORKER_BASE_COUNT

    _WORKER_TASKS = {int(task.task_index): task for task in tasks}
    _WORKER_BANK = training_bank
    _WORKER_REPOSITORY = repository
    _WORKER_CFG = cfg
    _WORKER_BASE_COUNT = int(base_count)


def build_worker(task_index: int) -> ExpandableAttackModel:
    if _WORKER_BANK is None:
        raise RuntimeError("Worker training bank is not initialized.")
    return build_initial_attack_model(
        base_task=_WORKER_TASKS[int(task_index)],
        training_bank=_WORKER_BANK,
        repository=_WORKER_REPOSITORY,
        base_count=_WORKER_BASE_COUNT,
        cfg=_WORKER_CFG,
    )


def cache_path(dataset: str, cfg, base_count: int) -> Path:
    result_dir = (
        Path(__file__).resolve().parent
        / cfg.result_dir
        / "attack_cache"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset_safe = str(dataset).strip().replace("/", "_").replace("\\", "_")
    return result_dir / (
        f"{dataset_safe}_poolsize_attack_base{int(base_count)}_v{CACHE_VERSION}.pkl"
    )


def cache_metadata(
    dataset: str,
    cfg,
    tasks: Sequence[SLATask],
    base_count: int,
) -> Dict:
    node_path, service_path = resolve_dataset_paths(
        dataset,
        cfg.node_file,
        cfg.service_file,
    )
    return {
        "cache_version": CACHE_VERSION,
        "attack_definition": (
            "pool_size_aligned_expandable_training_empirical_hypothesis_stream_"
            "single_public_conventional_template"
        ),
        "dataset": str(dataset),
        "node_file": str(cfg.node_file),
        "service_file": str(cfg.service_file),
        "node_sha256": file_sha256(node_path),
        "service_sha256": file_sha256(service_path),
        "task_indices": [int(task.task_index) for task in tasks],
        "base_count": int(base_count),
        "ga_seed": int(cfg.ga_seed),
        "ga_population": int(cfg.ga_population),
        "ga_max_generations": int(cfg.ga_max_generations),
        "ga_stagnation_patience": int(cfg.ga_stagnation_patience),
        "ga_crossover_rate": float(cfg.ga_crossover_rate),
        "ga_mutation_probability": float(cfg.ga_mutation_probability),
        "ga_elite_count": int(cfg.ga_elite_count),
    }


def atomic_pickle(path: Path, payload) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def save_attack_cache(
    path: Path,
    metadata: Dict,
    models: Dict[int, ExpandableAttackModel],
    complete: bool,
) -> None:
    atomic_pickle(
        path,
        {
            "metadata": metadata,
            "models": models,
            "complete": bool(complete),
        },
    )


def load_attack_cache(
    path: Path,
    expected_metadata: Dict,
) -> Tuple[Dict[int, ExpandableAttackModel], bool]:
    if not path.is_file():
        return {}, False
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(f"Attack cache ignored: {exc}", flush=True)
        return {}, False

    if payload.get("metadata") != expected_metadata:
        print("Attack cache ignored because metadata changed.", flush=True)
        return {}, False

    models = {
        int(task_index): model
        for task_index, model in payload.get("models", {}).items()
    }
    return models, bool(payload.get("complete", False))


def build_attack_database(
    dataset: str,
    tasks: Sequence[SLATask],
    training_bank: TrainingRequirementBank,
    repository: Dict[int, Tuple[Service, ...]],
    base_count: int,
    cfg,
    workers: int,
    rebuild_cache: bool,
) -> Tuple[
    Dict[int, ExpandableAttackModel],
    Path,
    Dict,
    bool,
]:
    tasks = list(tasks)
    workers = max(1, int(workers))
    path = cache_path(dataset, cfg, base_count)
    metadata = cache_metadata(dataset, cfg, tasks, base_count)

    if rebuild_cache:
        models: Dict[int, ExpandableAttackModel] = {}
        complete = False
    else:
        models, complete = load_attack_cache(path, metadata)

    expected = {int(task.task_index) for task in tasks}
    models = {
        int(task_index): model
        for task_index, model in models.items()
        if int(task_index) in expected
    }

    if complete and set(models) == expected:
        return models, path, metadata, True

    missing = [
        task
        for task in tasks
        if int(task.task_index) not in models
    ]

    def save_partial(is_complete: bool) -> None:
        save_attack_cache(path, metadata, models, is_complete)

    since_save = 0
    if workers <= 1:
        for task in tqdm(missing, desc="Build initial public attack bank"):
            models[int(task.task_index)] = build_initial_attack_model(
                base_task=task,
                training_bank=training_bank,
                repository=repository,
                base_count=base_count,
                cfg=cfg,
            )
            since_save += 1
            if since_save >= 10:
                save_partial(False)
                since_save = 0
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(
                missing,
                training_bank,
                repository,
                cfg,
                base_count,
            ),
        ) as executor:
            futures = {
                executor.submit(build_worker, int(task.task_index)): int(task.task_index)
                for task in missing
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Build initial public attack bank ({workers} workers)",
            ):
                task_index = futures[future]
                models[task_index] = future.result()
                since_save += 1
                if since_save >= 10:
                    save_partial(False)
                    since_save = 0

    if set(models) != expected:
        missing_ids = sorted(expected - set(models))
        raise RuntimeError(
            "Initial attack-bank construction incomplete. Missing task IDs: "
            f"{missing_ids[:10]}"
        )

    save_partial(True)
    return models, path, metadata, False


def public_templates(
    attack_model: ExpandableAttackModel,
    hypothesis_count: int,
) -> Tuple[AttackTemplate, ...]:
    hypothesis_count = int(hypothesis_count)
    if hypothesis_count < 2:
        raise ValueError("hypothesis_count must be >= 2.")
    if len(attack_model.templates) < hypothesis_count:
        raise RuntimeError(
            f"Task {attack_model.task_index}: only {len(attack_model.templates)} "
            f"templates are available for K={hypothesis_count}."
        )

    templates = tuple(attack_model.templates[:hypothesis_count])
    if not any(
        same_hypothesis(template.hypothesis, attack_model.true_hypothesis)
        for template in templates
    ):
        raise RuntimeError(
            f"Task {attack_model.task_index}: true hypothesis is absent from "
            f"the K={hypothesis_count} closed set."
        )
    return templates


def normalized_hamming_distance(
    left: Sequence[ServiceIdentity],
    right: Sequence[ServiceIdentity],
) -> float:
    if len(left) != len(right):
        raise ValueError("Composition sequences must have the same length.")
    if not left:
        raise ValueError("Composition sequences cannot be empty.")
    mismatches = sum(int(a != b) for a, b in zip(left, right))
    return float(mismatches / len(left))


def jaccard_service_distance(
    left: Sequence[ServiceIdentity],
    right: Sequence[ServiceIdentity],
) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return float(1.0 - len(left_set & right_set) / len(union))


def qos_euclidean_distance(
    observation: ReleaseObservation,
    template: AttackTemplate,
) -> float:
    return float(
        math.sqrt(
            (float(observation.q2_product) - float(template.q2_product)) ** 2
            + (float(observation.q3_product) - float(template.q3_product)) ** 2
        )
    )


def template_distance(
    observation: ReleaseObservation,
    template: AttackTemplate,
    distance_name: str,
) -> float:
    distance_name = str(distance_name).lower()
    if distance_name == "hamming":
        return normalized_hamming_distance(
            observation.service_ids,
            template.service_ids,
        )
    if distance_name == "jaccard":
        return jaccard_service_distance(
            observation.service_ids,
            template.service_ids,
        )
    if distance_name == "qos":
        return qos_euclidean_distance(observation, template)
    raise ValueError(f"Unsupported attack distance: {distance_name}")


def blackbox_infer_from_release(
    observation: ReleaseObservation,
    templates: Sequence[AttackTemplate],
    distance_name: str,
) -> InferenceResult:

    if not templates:
        raise RuntimeError("Public attack template set is empty.")
    if distance_name not in SUPPORTED_DISTANCES:
        raise ValueError(f"Unsupported attack distance: {distance_name}")

    distances = tuple(
        float(template_distance(observation, template, distance_name))
        for template in templates
    )
    best_distance = min(distances)
    winners = tuple(
        index
        for index, value in enumerate(distances)
        if abs(value - best_distance) <= DIST_TOL
    )
    if not winners:
        raise RuntimeError("Internal error: no attack winner.")

    return InferenceResult(
        winner_indices=winners,
        distances=distances,
    )

def true_template_index(
    attack_model: ExpandableAttackModel,
    templates: Sequence[AttackTemplate],
) -> int:
    matches = [
        index
        for index, template in enumerate(templates)
        if same_hypothesis(template.hypothesis, attack_model.true_hypothesis)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Task {attack_model.task_index}: expected one true index in the "
            f"current hypothesis set, got {matches}."
        )
    return int(matches[0])


def evaluate_inference(
    attack_model: ExpandableAttackModel,
    templates: Sequence[AttackTemplate],
    inference: InferenceResult,
    release_pool_size: int,
    representative_count: int,
) -> Dict[str, float]:

    K = len(templates)
    if K < 2:
        raise RuntimeError("Attack hypothesis set must contain at least 2 profiles.")
    if len(inference.distances) != K:
        raise RuntimeError("Inference score count does not match hypothesis count.")

    truth_index = true_template_index(attack_model, templates)
    winners = inference.winner_indices

    accuracy_probability = (
        1.0 / len(winners)
        if truth_index in winners
        else 0.0
    )
    accuracy_percent = 100.0 * accuracy_probability
    random_probability = 1.0 / K
    random_accuracy_percent = 100.0 * random_probability

    chance_normalized_advantage = (
        (accuracy_probability - random_probability)
        / (1.0 - random_probability)
    )

    truth = attack_model.true_hypothesis
    constraint_nmae = statistics.mean(
        requirement_profile_nmae(
            estimate=templates[index].hypothesis,
            truth=truth,
        )
        for index in winners
    )

    random_guess_constraint_nmae = statistics.mean(
        requirement_profile_nmae(
            estimate=template.hypothesis,
            truth=truth,
        )
        for template in templates
    )

    true_distance = float(inference.distances[truth_index])
    strictly_better = sum(
        value < true_distance - DIST_TOL
        for value in inference.distances
    )
    tied_with_truth = sum(
        abs(value - true_distance) <= DIST_TOL
        for value in inference.distances
    )
    if tied_with_truth <= 0:
        raise RuntimeError("Internal error: true hypothesis has no rank tie group.")

    expected_rank = (
        strictly_better
        + (tied_with_truth + 1.0) / 2.0
    )
    true_rank_percentile = (expected_rank - 1.0) / (K - 1.0)

    return {
        "release_pool_size": float(release_pool_size),
        "attack_hypothesis_count": float(K),
        "representative_count": float(representative_count),
        "random_guess_accuracy_percent": float(random_accuracy_percent),
        "attack_accuracy_percent": float(accuracy_percent),
        "chance_normalized_recovery_advantage": float(
            chance_normalized_advantage
        ),
        "constraint_nmae": float(constraint_nmae),
        "random_guess_constraint_nmae": float(
            random_guess_constraint_nmae
        ),
        "true_rank_percentile": float(true_rank_percentile),
    }


def selected_candidate_observation(
    selected: SequenceCandidate,
    task: SLATask,
) -> ReleaseObservation:
    chromosome = tuple(int(gene) for gene in selected.chromosome)
    if len(chromosome) != len(task.services):
        raise RuntimeError(
            f"Task {task.task_index}: selected chromosome length mismatch."
        )

    identities: List[ServiceIdentity] = []
    for position, gene in enumerate(chromosome):
        if not 0 <= gene < len(task.services[position]):
            raise RuntimeError(
                f"Task {task.task_index}, position {position}: selected gene "
                f"{gene} is outside the victim task."
            )
        service = task.services[position][gene]
        identities.append((int(service[4]), int(service[5])))

    return ReleaseObservation(
        service_ids=tuple(identities),
        q2_product=float(selected.evaluation.q2_product),
        q3_product=float(selected.evaluation.q3_product),
    )


def attack_observation(
    attack_model: ExpandableAttackModel,
    observation: ReleaseObservation,
    hypothesis_count: int,
    distance_name: str,
    release_pool_size: int,
    representative_count: int,
) -> Dict[str, float]:
    templates = public_templates(attack_model, hypothesis_count)
    inference = blackbox_infer_from_release(
        observation=observation,
        templates=templates,
        distance_name=distance_name,
    )
    return evaluate_inference(
        attack_model=attack_model,
        templates=templates,
        inference=inference,
        release_pool_size=release_pool_size,
        representative_count=representative_count,
    )


def attack_selected_candidate(
    attack_model: ExpandableAttackModel,
    selected: SequenceCandidate,
    victim_task: SLATask,
    hypothesis_count: int,
    distance_name: str,
    release_pool_size: int,
    representative_count: int,
) -> Dict[str, float]:
    return attack_observation(
        attack_model=attack_model,
        observation=selected_candidate_observation(selected, victim_task),
        hypothesis_count=hypothesis_count,
        distance_name=distance_name,
        release_pool_size=release_pool_size,
        representative_count=representative_count,
    )


def conventional_attack_metrics(
    attack_model: ExpandableAttackModel,
    hypothesis_count: int,
    distance_name: str,
    nominal_release_pool_size: int,
    representative_count: int,
) -> Dict[str, float]:
    true_template = attack_model.true_template()
    observation = ReleaseObservation(
        service_ids=true_template.service_ids,
        q2_product=true_template.q2_product,
        q3_product=true_template.q3_product,
    )
    return attack_observation(
        attack_model=attack_model,
        observation=observation,
        hypothesis_count=hypothesis_count,
        distance_name=distance_name,
        release_pool_size=nominal_release_pool_size,
        representative_count=representative_count,
    )


def delta_pool_metrics(pool, delta_value: float) -> Dict[str, float]:
    delta_value = float(delta_value)
    if not 0.0 <= delta_value < 1.0:
        raise ValueError("Delta ablation requires 0 <= delta < 1.")

    reference_utility = (
        float(pool.minimum_expected_utility)
        / (1.0 - delta_value)
    )
    expected_utility = float(pool.expected_utility)
    if reference_utility <= 1e-15:
        actual_degradation_percent = 0.0
    else:
        actual_degradation_percent = 100.0 * max(
            0.0,
            1.0 - expected_utility / reference_utility,
        )

    if actual_degradation_percent > 100.0 * delta_value + 1e-8:
        raise RuntimeError(
            "Phase-1 utility degradation exceeds configured delta: "
            f"actual={actual_degradation_percent:.10f}% "
            f"delta={100.0 * delta_value:.10f}%"
        )

    return {
        "pool_size": float(len(pool.candidates)),
        "representative_count": float(pool.representative_count),
        "representativeness": float(pool.representativeness),
        "phase1_expected_utility": float(expected_utility),
        "phase1_reference_utility": float(reference_utility),
        "phase1_actual_utility_degradation_percent": float(
            actual_degradation_percent
        ),
    }


def target_tasks_from_config(
    all_tasks: Sequence[SLATask],
    cfg,
    task_limit: Optional[int],
) -> List[SLATask]:
    tasks = list(all_tasks)

    exact_max = getattr(cfg, "exact_max_tasks", None)
    if exact_max is not None:
        tasks = tasks[: int(exact_max)]

    ga_max = getattr(cfg, "ga_max_exact_tasks", None)
    if ga_max is not None:
        tasks = tasks[: int(ga_max)]

    if task_limit is not None:
        tasks = tasks[: int(task_limit)]

    return tasks


def dataset_proxy_pool_size(para: Dict, dataset: str) -> int:
    if "proxy_pool_size" in para:
        return int(para["proxy_pool_size"])

    dataset_settings = para.get("dataset_settings", {})
    current = dataset_settings.get(str(dataset), {})
    if "proxy_pool_size" not in current:
        raise KeyError("Cannot resolve proxy_pool_size from dp_para.")
    return int(current["proxy_pool_size"])


def choose_grid(
    mechanisms_filter: Optional[Sequence[str]],
    epsilons_filter: Optional[Sequence[float]],
    clip_bounds_filter: Optional[Sequence[float]],
) -> Tuple[List[str], List[float], List[float]]:
    mechanisms = [str(value) for value in dp_para["noise_types"]]
    epsilon_values = [float(value) for value in dp_para["epsilon"]]
    clip_bounds = [float(value) for value in dp_para["clip_bounds"]]

    if mechanisms_filter is not None:
        wanted = {str(value) for value in mechanisms_filter}
        mechanisms = [value for value in mechanisms if value in wanted]
    if epsilons_filter is not None:
        wanted = {float(value) for value in epsilons_filter}
        epsilon_values = [value for value in epsilon_values if value in wanted]
    if clip_bounds_filter is not None:
        wanted = {float(value) for value in clip_bounds_filter}
        clip_bounds = [value for value in clip_bounds if value in wanted]

    if not mechanisms:
        raise ValueError("No mechanism remains after filtering.")
    if not epsilon_values:
        raise ValueError("No epsilon remains after filtering.")
    if not clip_bounds:
        raise ValueError("No clip bound remains after filtering.")

    unsupported = set(mechanisms) - {"PNF", "EM"}
    if unsupported:
        raise ValueError(f"Unsupported mechanisms: {sorted(unsupported)}")

    return mechanisms, epsilon_values, clip_bounds


def choose_deltas(
    delta_values_filter: Optional[Sequence[float]],
) -> List[float]:
    configured = dp_para.get("delta_ablation_values", DEFAULT_DELTA_VALUES)
    values = (
        [float(value) for value in configured]
        if delta_values_filter is None
        else [float(value) for value in delta_values_filter]
    )
    if not values:
        raise ValueError("At least one delta value is required.")
    if any(not 0.0 <= value < 1.0 for value in values):
        raise ValueError("Every delta value must satisfy 0 <= delta < 1.")
    return values


def choose_distances(
    distance_filter: Optional[Sequence[str]],
) -> List[str]:
    values = (
        list(SUPPORTED_DISTANCES)
        if distance_filter is None
        else [str(value).lower() for value in distance_filter]
    )
    if not values:
        raise ValueError("At least one attack distance is required.")
    unsupported = set(values) - set(SUPPORTED_DISTANCES)
    if unsupported:
        raise ValueError(f"Unsupported attack distances: {sorted(unsupported)}")
    # Stable de-duplication.
    return list(dict.fromkeys(values))


def method_order(method: str) -> int:
    return {
        "Conventional_GA": 0,
        "SamePool_Non_DP": 1,
        "PNF": 2,
        "EM": 3,
    }.get(str(method), 99)


# =============================================================================
# Output
# =============================================================================
def result_path(dataset: str, cfg) -> Path:
    dataset_safe = str(dataset).strip().replace("/", "_").replace("\\", "_")
    result_dir = Path(__file__).resolve().parent / cfg.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / (
        f"{dataset_safe}_sequence_dp_poolsize_representative_inference_attack.txt"
    )


def atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{float(mean_value):.6f} ± {float(std_value):.6f}"


def metric_cell(stats: RunningMetricStats, key: str) -> str:
    mean_value, std_value = stats.mean_std(key)
    return format_mean_std(mean_value, std_value)


def write_attack_table_header(lines: List[str]) -> None:
    lines.append(
        "\t".join(
            [
                "PoolVariant",
                "Delta",
                "Distance",
                "ReleaseMethod",
                "Epsilon",
                "ClipBound",
                "ReleasePoolSize(Mean±Std)",
                "AttackHypothesisCount(Mean±Std)",
                "RepresentativeCount(Mean±Std)",
                "RandomGuessAccuracy%(Mean±Std)",
                "HypothesisRecoveryAccuracy%(Mean±Std)",
                "ChanceNormalizedRecoveryAdvantage(Mean±Std)",
                "ConstraintNMAE(Mean±Std)",
                "RandomGuessConstraintNMAE(Mean±Std)",
                "TrueRankPercentile(Mean±Std)",
                "PreferenceWeightMAE",
            ]
        )
    )


def write_attack_row(
    lines: List[str],
    pool_variant: str,
    delta: Optional[float],
    distance_name: str,
    method: str,
    epsilon: Optional[float],
    clip_bound: Optional[float],
    stats: RunningMetricStats,
    completed_runs: int,
) -> None:
    if stats.count != completed_runs:
        raise RuntimeError(
            f"Setting {(pool_variant, delta, distance_name, method, epsilon, clip_bound)} "
            f"has {stats.count} runs; expected {completed_runs}."
        )

    lines.append(
        "\t".join(
            [
                str(pool_variant),
                "NA" if delta is None else f"{float(delta):.6f}",
                str(distance_name),
                str(method),
                "NA" if epsilon is None else f"{float(epsilon):.6f}",
                "NA" if clip_bound is None else f"{float(clip_bound):.6f}",
                metric_cell(stats, "release_pool_size"),
                metric_cell(stats, "attack_hypothesis_count"),
                metric_cell(stats, "representative_count"),
                metric_cell(stats, "random_guess_accuracy_percent"),
                metric_cell(stats, "attack_accuracy_percent"),
                metric_cell(stats, "chance_normalized_recovery_advantage"),
                metric_cell(stats, "constraint_nmae"),
                metric_cell(stats, "random_guess_constraint_nmae"),
                metric_cell(stats, "true_rank_percentile"),
                "NA",
            ]
        )
    )


def write_results(
    output_path: Path,
    dataset: str,
    task_count: int,
    completed_runs: int,
    total_runs: int,
    optimal_count: int,
    proxy_pool_size: int,
    delta_values: Sequence[float],
    mechanisms: Sequence[str],
    epsilon_values: Sequence[float],
    clip_bounds: Sequence[float],
    distances: Sequence[str],
    aligned_stats: Dict[AlignedSettingKey, RunningMetricStats],
    matched_stats: Dict[MatchedSettingKey, RunningMetricStats],
    pool_stats: Dict[float, RunningMetricStats],
    training_bank: TrainingRequirementBank,
    matched_control_enabled: bool,
) -> None:
    sample_counts = [
        len(values)
        for values in training_bank.samples_by_category.values()
    ]
    unique_counts = [
        len(values)
        for values in training_bank.unique_by_category.values()
    ]

    lines = [
        f"Dataset: {dataset}",
        f"Tasks: {int(task_count)}",
        f"TotalPipelineRuns: {int(total_runs)}",
        f"CompletedRuns: {int(completed_runs)}",
        f"OptimalCandidateCount: {int(optimal_count)}",
        f"ProxyPoolSize: {int(proxy_pool_size)}",
        "DeltaValues: " + ",".join(f"{float(v):g}" for v in delta_values),
        "EpsilonValues: " + ",".join(f"{float(v):g}" for v in epsilon_values),
        "ClipBounds: " + ",".join(f"{float(v):g}" for v in clip_bounds),
        "AttackDistances: " + ",".join(str(v) for v in distances),
        (
            "AttackModel: manuscript Section-V black-box rule; for every public "
            "requirement hypothesis H_j run the public conventional SLA-GA once "
            "to obtain CS*(H_j), then infer argmin_j d(CS*(H_j), CS_obs)"
        ),
        (
            "PoolAlignedHypothesisRule: for each task/run/pool, K = |Psi_release|; "
            "the experimenter uses only this SIZE to choose K public requirement "
            "hypotheses. The attacker never receives any private pool member."
        ),
        (
            "HypothesisSource: first-75%-training empirical local QoS requirement "
            "bank; every local tuple is an observed same-category training tuple; "
            "templates are generated independently of release/pool identities."
        ),
        (
            "OnlineAttackerInput: final released composition + K public "
            "requirement/templates + predeclared distance ONLY"
        ),
        (
            "ForbiddenOnlineInputs: evaluator true label, private candidate source, "
            "Psi_opt/Psi_rep members, utilities, clipped utilities, epsilon, tau, "
            "mechanism state, PNF/EM randomness"
        ),
        (
            "PaperMetric1: Hypothesis Recovery Accuracy; lower is better privacy"
        ),
        (
            "PaperMetric2: QoS Constraint Recovery NMAE over all 4*T normalized "
            "local requirement endpoints; higher is better privacy"
        ),
        (
            "PaperMetric3: Preference Weight MAE = NA because the current "
            "Sequence-DP implementation has no user-specific preference-weight p"
        ),
        (
            "AddedMetric1: ChanceNormalizedRecoveryAdvantage = "
            "(Acc-1/K)/(1-1/K); 0 is random exact-recovery level, 1 is perfect "
            "recovery; lower is better privacy"
        ),
        (
            "AddedMetric2: TrueRankPercentile = (E[rank_true]-1)/(K-1), with "
            "uniform expected rank inside exact-distance ties; 0 is rank-1 "
            "leakage, random ranking has expectation 0.5; higher is better privacy"
        ),
        (
            "MatchedKControl: OPT_ONLY release is additionally attacked with the "
            "same K used for the corresponding REPRESENTATIVE pool, while still "
            "hiding all pool members. This isolates release-support diversification "
            "from the trivial effect of enlarging the hypothesis count."
        ),
        (
            "TrainingCategorySampleCountRange: "
            f"{min(sample_counts)}..{max(sample_counts)}"
        ),
        (
            "TrainingCategoryUniqueCountRange: "
            f"{min(unique_counts)}..{max(unique_counts)}"
        ),
        "",
        "[RepresentativePoolDiagnostics - evaluator only, NEVER used by online attack]",
        "\t".join(
            [
                "Delta",
                "PoolSize(Mean±Std)",
                "RepresentativeCount(Mean±Std)",
                "Representativeness(Mean±Std)",
                "Phase1ExpectedUtility(Mean±Std)",
                "Phase1ReferenceUtility(Mean±Std)",
                "ActualUtilityDegradation%(Mean±Std)",
            ]
        ),
    ]

    for delta in sorted(pool_stats):
        stats = pool_stats[delta]
        if stats.count != completed_runs:
            raise RuntimeError(
                f"Pool delta={delta} has {stats.count} runs; expected {completed_runs}."
            )
        lines.append(
            "\t".join(
                [
                    f"{float(delta):.6f}",
                    metric_cell(stats, "pool_size"),
                    metric_cell(stats, "representative_count"),
                    metric_cell(stats, "representativeness"),
                    metric_cell(stats, "phase1_expected_utility"),
                    metric_cell(stats, "phase1_reference_utility"),
                    metric_cell(stats, "phase1_actual_utility_degradation_percent"),
                ]
            )
        )

    lines.extend(["", "[PoolAlignedPaperAttackMetrics]"])
    write_attack_table_header(lines)

    def aligned_sort_key(item):
        key, _ = item
        variant, delta, distance_name, method, epsilon, clip_bound = key
        return (
            0 if variant == "OPT_ONLY" else 1,
            -1.0 if delta is None else float(delta),
            SUPPORTED_DISTANCES.index(distance_name),
            method_order(method),
            -1.0 if epsilon is None else float(epsilon),
            -1.0 if clip_bound is None else float(clip_bound),
        )

    for key, stats in sorted(aligned_stats.items(), key=aligned_sort_key):
        pool_variant, delta, distance_name, method, epsilon, clip_bound = key
        write_attack_row(
            lines=lines,
            pool_variant=pool_variant,
            delta=delta,
            distance_name=distance_name,
            method=method,
            epsilon=epsilon,
            clip_bound=clip_bound,
            stats=stats,
            completed_runs=completed_runs,
        )

    if matched_control_enabled:
        lines.extend(
            [
                "",
                "[MatchedHypothesisCountControl - OPT_ONLY release, K matched to REPRESENTATIVE pool]",
            ]
        )
        write_attack_table_header(lines)

        def matched_sort_key(item):
            key, _ = item
            delta, distance_name, method, epsilon, clip_bound = key
            return (
                float(delta),
                SUPPORTED_DISTANCES.index(distance_name),
                method_order(method),
                -1.0 if epsilon is None else float(epsilon),
                -1.0 if clip_bound is None else float(clip_bound),
            )

        for key, stats in sorted(matched_stats.items(), key=matched_sort_key):
            delta, distance_name, method, epsilon, clip_bound = key
            write_attack_row(
                lines=lines,
                pool_variant="OPT_ONLY_MATCHED_K",
                delta=delta,
                distance_name=distance_name,
                method=method,
                epsilon=epsilon,
                clip_bound=clip_bound,
                stats=stats,
                completed_runs=completed_runs,
            )

    atomic_write_text(output_path, "\n".join(lines) + "\n")


# =============================================================================
# Main experiment
# =============================================================================
def run_poolsize_representative_attack(
    dataset: str,
    workers: int,
    task_limit: Optional[int],
    rebuild_cache: bool,
    validate_only: bool,
    mechanisms_filter: Optional[Sequence[str]],
    epsilons_filter: Optional[Sequence[float]],
    clip_bounds_filter: Optional[Sequence[float]],
    delta_values_filter: Optional[Sequence[float]],
    distances_filter: Optional[Sequence[str]],
    runs_override: Optional[int],
    matched_control_enabled: bool,
) -> None:
    validate_dp_para(dp_para)
    cfg = load_config(dataset=dataset)

    if not cfg.test_only:
        raise RuntimeError(
            "This attack requires TEST_ONLY=True so the first 75% training split "
            "is attacker background and held-out workflows are evaluation contexts."
        )

    all_tasks = load_sla_tasks(
        dataset,
        cfg.node_file,
        cfg.service_file,
        test_only=cfg.test_only,
        min_candidates=1,
    )
    target_tasks = target_tasks_from_config(
        all_tasks=all_tasks,
        cfg=cfg,
        task_limit=task_limit,
    )
    if not target_tasks:
        raise RuntimeError("No held-out workflow context is available.")

    repository = load_public_service_repository(
        dataset=dataset,
        node_filename=cfg.node_file,
        service_filename=cfg.service_file,
    )
    training_bank = load_training_requirement_bank(
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
            "Training split lacks local requirement observations for public "
            f"categories: {missing_categories}"
        )

    mechanisms, epsilon_values, clip_bounds = choose_grid(
        mechanisms_filter,
        epsilons_filter,
        clip_bounds_filter,
    )
    delta_values = choose_deltas(delta_values_filter)
    distances = choose_distances(distances_filter)

    optimal_count = int(dp_para["optimal_candidate_count"])
    proxy_pool_size = dataset_proxy_pool_size(dp_para, dataset)
    pipeline_runs = (
        int(dp_para["pipeline_runs"])
        if runs_override is None
        else int(runs_override)
    )
    if pipeline_runs <= 0:
        raise ValueError("pipeline run count must be positive.")

    print("============================================================")
    print("POOL-SIZE-ALIGNED BLACK-BOX REQUIREMENT INFERENCE ATTACK")
    print("============================================================")
    print(f"Dataset                    : {dataset}")
    print(f"Tasks                      : {len(target_tasks)}")
    print(f"Pipeline runs              : {pipeline_runs}")
    print(f"Psi_opt size               : {optimal_count}")
    print(f"Proxy source size          : {proxy_pool_size}")
    print(f"Deltas                     : {delta_values}")
    print(f"Mechanisms                 : {mechanisms}")
    print(f"Epsilons                   : {epsilon_values}")
    print(f"Clip bounds                : {clip_bounds}")
    print(f"Attack distances           : {distances}")
    print(
        "Hypothesis rule            : K = release pool size; K public QoS "
        "profiles, NOT private pool members"
    )
    print(
        "Online attacker            : CS_obs + public K templates + distance only"
    )
    print(
        f"Matched-K control          : {'ON' if matched_control_enabled else 'OFF'}"
    )
    print("============================================================\n")

    precompute_start = time.time()
    (
        attack_database,
        attack_cache,
        attack_cache_metadata,
        cache_hit,
    ) = build_attack_database(
        dataset=dataset,
        tasks=target_tasks,
        training_bank=training_bank,
        repository=repository,
        base_count=optimal_count,
        cfg=cfg,
        workers=workers,
        rebuild_cache=rebuild_cache,
    )
    print(
        "Initial public attack bank ready | "
        f"baseK={optimal_count} | "
        f"cache={'HIT' if cache_hit else 'BUILT/UPDATED'} | "
        f"elapsed={time.time() - precompute_start:.2f}s"
    )
    print(f"Attack cache                : {attack_cache}")

    if validate_only:
        print(
            "Validation-only mode finished after building the minimum K=|Psi_opt| "
            "public hypothesis/template bank."
        )
        return

    output_path = result_path(dataset, cfg)

    aligned_stats: Dict[AlignedSettingKey, RunningMetricStats] = {}
    matched_stats: Dict[MatchedSettingKey, RunningMetricStats] = {}
    pool_stats: Dict[float, RunningMetricStats] = {}

    total_start = time.time()

    for run_index in range(1, pipeline_runs + 1):
        run_start = time.time()
        run_aligned: Dict[AlignedSettingKey, MetricAccumulator] = {}
        run_matched: Dict[MatchedSettingKey, MetricAccumulator] = {}
        run_pool: Dict[float, MetricAccumulator] = {
            float(delta): MetricAccumulator(POOL_METRIC_KEYS)
            for delta in delta_values
        }

        expanded_templates_this_run = 0
        tasks_since_cache_save = 0

        for base_task in tqdm(
            target_tasks,
            desc=f"Pool-size attack {run_index}/{pipeline_runs}",
        ):
            task_index = int(base_task.task_index)
            attack_model = attack_database[task_index]

            victim_task = make_hypothetical_task(
                base_task=base_task,
                hypothesis=attack_model.true_hypothesis,
                repository=repository,
            )

            source_seed = (
                int(cfg.ga_seed)
                + int(run_index) * 10_000_019
                + task_index
            )
            source = build_candidate_source(
                task=victim_task,
                cfg=cfg,
                seed=source_seed,
                optimal_count=optimal_count,
                proxy_pool_size=proxy_pool_size,
            )

            opt_candidates = list(source.quality_candidates[:optimal_count])
            if len(opt_candidates) != optimal_count:
                raise RuntimeError(
                    f"Task {task_index}: expected {optimal_count} Psi_opt "
                    f"candidates, got {len(opt_candidates)}."
                )
            opt_pool = PoolView(candidates=opt_candidates)
            opt_k = len(opt_pool.candidates)

            representative_pools = {}
            max_required_k = opt_k
            for delta_value in delta_values:
                para_delta = dict(dp_para)
                para_delta["max_utility_degradation"] = float(delta_value)
                validate_dp_para(para_delta)
                pool = build_representative_pool(
                    source=source,
                    optimal_count=optimal_count,
                    para=para_delta,
                )
                if len(pool.candidates) != optimal_count + int(pool.representative_count):
                    raise RuntimeError(
                        f"Task {task_index}, delta={delta_value}: pool size "
                        "does not equal optimal_count + representative_count."
                    )
                representative_pools[float(delta_value)] = pool
                max_required_k = max(max_required_k, len(pool.candidates))
                run_pool[float(delta_value)].add(
                    delta_pool_metrics(pool, float(delta_value))
                )

            added = ensure_attack_template_count(
                attack_model=attack_model,
                target_count=max_required_k,
                base_task=base_task,
                training_bank=training_bank,
                repository=repository,
                cfg=cfg,
            )
            expanded_templates_this_run += added
            tasks_since_cache_save += 1

            if (
                expanded_templates_this_run > 0
                and tasks_since_cache_save >= CACHE_SAVE_EVERY_TASKS
            ):
                save_attack_cache(
                    attack_cache,
                    attack_cache_metadata,
                    attack_database,
                    True,
                )
                tasks_since_cache_save = 0

            true_template = attack_model.true_template()
            conventional_observation = ReleaseObservation(
                service_ids=true_template.service_ids,
                q2_product=true_template.q2_product,
                q3_product=true_template.q3_product,
            )

            opt_non_dp = select_candidate(
                pool=opt_pool,
                mechanism="Non_DP",
            )
            opt_non_dp_observation = selected_candidate_observation(
                opt_non_dp,
                victim_task,
            )

            for distance_name in distances:
                key: AlignedSettingKey = (
                    "OPT_ONLY",
                    None,
                    distance_name,
                    "Conventional_GA",
                    None,
                    None,
                )
                run_aligned.setdefault(
                    key,
                    MetricAccumulator(ATTACK_METRIC_KEYS),
                ).add(
                    attack_observation(
                        attack_model=attack_model,
                        observation=conventional_observation,
                        hypothesis_count=opt_k,
                        distance_name=distance_name,
                        release_pool_size=opt_k,
                        representative_count=0,
                    )
                )

                key = (
                    "OPT_ONLY",
                    None,
                    distance_name,
                    "SamePool_Non_DP",
                    None,
                    None,
                )
                run_aligned.setdefault(
                    key,
                    MetricAccumulator(ATTACK_METRIC_KEYS),
                ).add(
                    attack_observation(
                        attack_model=attack_model,
                        observation=opt_non_dp_observation,
                        hypothesis_count=opt_k,
                        distance_name=distance_name,
                        release_pool_size=opt_k,
                        representative_count=0,
                    )
                )

            representative_release_observations: Dict[
                Tuple[float, str, Optional[float], Optional[float]],
                ReleaseObservation,
            ] = {}

            for delta_value in delta_values:
                delta_value = float(delta_value)
                pool = representative_pools[delta_value]
                rep_k = len(pool.candidates)
                rep_count = int(pool.representative_count)

                rep_non_dp = select_candidate(
                    pool=pool,
                    mechanism="Non_DP",
                )
                rep_non_dp_observation = selected_candidate_observation(
                    rep_non_dp,
                    victim_task,
                )
                representative_release_observations[
                    (delta_value, "SamePool_Non_DP", None, None)
                ] = rep_non_dp_observation

                for distance_name in distances:
                    key = (
                        "REPRESENTATIVE",
                        delta_value,
                        distance_name,
                        "Conventional_GA",
                        None,
                        None,
                    )
                    run_aligned.setdefault(
                        key,
                        MetricAccumulator(ATTACK_METRIC_KEYS),
                    ).add(
                        attack_observation(
                            attack_model=attack_model,
                            observation=conventional_observation,
                            hypothesis_count=rep_k,
                            distance_name=distance_name,
                            release_pool_size=rep_k,
                            representative_count=rep_count,
                        )
                    )

                    key = (
                        "REPRESENTATIVE",
                        delta_value,
                        distance_name,
                        "SamePool_Non_DP",
                        None,
                        None,
                    )
                    run_aligned.setdefault(
                        key,
                        MetricAccumulator(ATTACK_METRIC_KEYS),
                    ).add(
                        attack_observation(
                            attack_model=attack_model,
                            observation=rep_non_dp_observation,
                            hypothesis_count=rep_k,
                            distance_name=distance_name,
                            release_pool_size=rep_k,
                            representative_count=rep_count,
                        )
                    )

                for epsilon in epsilon_values:
                    for clip_bound in clip_bounds:
                        for mechanism in mechanisms:
                            selected = select_candidate(
                                pool=pool,
                                mechanism=mechanism,
                                epsilon=float(epsilon),
                                clip_bound=float(clip_bound),
                            )
                            observation = selected_candidate_observation(
                                selected,
                                victim_task,
                            )
                            representative_release_observations[
                                (
                                    delta_value,
                                    mechanism,
                                    float(epsilon),
                                    float(clip_bound),
                                )
                            ] = observation

                            for distance_name in distances:
                                key = (
                                    "REPRESENTATIVE",
                                    delta_value,
                                    distance_name,
                                    mechanism,
                                    float(epsilon),
                                    float(clip_bound),
                                )
                                run_aligned.setdefault(
                                    key,
                                    MetricAccumulator(ATTACK_METRIC_KEYS),
                                ).add(
                                    attack_observation(
                                        attack_model=attack_model,
                                        observation=observation,
                                        hypothesis_count=rep_k,
                                        distance_name=distance_name,
                                        release_pool_size=rep_k,
                                        representative_count=rep_count,
                                    )
                                )

            opt_random_observations: Dict[
                Tuple[str, float, float],
                ReleaseObservation,
            ] = {}

            for epsilon in epsilon_values:
                for clip_bound in clip_bounds:
                    for mechanism in mechanisms:
                        selected = select_candidate(
                            pool=opt_pool,
                            mechanism=mechanism,
                            epsilon=float(epsilon),
                            clip_bound=float(clip_bound),
                        )
                        observation = selected_candidate_observation(
                            selected,
                            victim_task,
                        )
                        opt_random_observations[
                            (mechanism, float(epsilon), float(clip_bound))
                        ] = observation

                        # Main pool-aligned attack: K = |Psi_opt|.
                        for distance_name in distances:
                            key = (
                                "OPT_ONLY",
                                None,
                                distance_name,
                                mechanism,
                                float(epsilon),
                                float(clip_bound),
                            )
                            run_aligned.setdefault(
                                key,
                                MetricAccumulator(ATTACK_METRIC_KEYS),
                            ).add(
                                attack_observation(
                                    attack_model=attack_model,
                                    observation=observation,
                                    hypothesis_count=opt_k,
                                    distance_name=distance_name,
                                    release_pool_size=opt_k,
                                    representative_count=0,
                                )
                            )

            if matched_control_enabled:
                for delta_value in delta_values:
                    delta_value = float(delta_value)
                    rep_pool = representative_pools[delta_value]
                    rep_k = len(rep_pool.candidates)

                    for distance_name in distances:
                        key_m: MatchedSettingKey = (
                            delta_value,
                            distance_name,
                            "SamePool_Non_DP",
                            None,
                            None,
                        )
                        run_matched.setdefault(
                            key_m,
                            MetricAccumulator(ATTACK_METRIC_KEYS),
                        ).add(
                            attack_observation(
                                attack_model=attack_model,
                                observation=opt_non_dp_observation,
                                hypothesis_count=rep_k,
                                distance_name=distance_name,
                                release_pool_size=opt_k,
                                representative_count=0,
                            )
                        )

                    for epsilon in epsilon_values:
                        for clip_bound in clip_bounds:
                            for mechanism in mechanisms:
                                observation = opt_random_observations[
                                    (
                                        mechanism,
                                        float(epsilon),
                                        float(clip_bound),
                                    )
                                ]
                                for distance_name in distances:
                                    key_m = (
                                        delta_value,
                                        distance_name,
                                        mechanism,
                                        float(epsilon),
                                        float(clip_bound),
                                    )
                                    run_matched.setdefault(
                                        key_m,
                                        MetricAccumulator(ATTACK_METRIC_KEYS),
                                    ).add(
                                        attack_observation(
                                            attack_model=attack_model,
                                            observation=observation,
                                            hypothesis_count=rep_k,
                                            distance_name=distance_name,
                                            release_pool_size=opt_k,
                                            representative_count=0,
                                        )
                                    )

        if expanded_templates_this_run > 0:
            save_attack_cache(
                attack_cache,
                attack_cache_metadata,
                attack_database,
                True,
            )

        for setting, accumulator in run_aligned.items():
            aligned_stats.setdefault(
                setting,
                RunningMetricStats(ATTACK_METRIC_KEYS),
            ).update(accumulator.mean())

        for setting, accumulator in run_matched.items():
            matched_stats.setdefault(
                setting,
                RunningMetricStats(ATTACK_METRIC_KEYS),
            ).update(accumulator.mean())

        for delta_value, accumulator in run_pool.items():
            pool_stats.setdefault(
                float(delta_value),
                RunningMetricStats(POOL_METRIC_KEYS),
            ).update(accumulator.mean())

        write_results(
            output_path=output_path,
            dataset=dataset,
            task_count=len(target_tasks),
            completed_runs=run_index,
            total_runs=pipeline_runs,
            optimal_count=optimal_count,
            proxy_pool_size=proxy_pool_size,
            delta_values=delta_values,
            mechanisms=mechanisms,
            epsilon_values=epsilon_values,
            clip_bounds=clip_bounds,
            distances=distances,
            aligned_stats=aligned_stats,
            matched_stats=matched_stats,
            pool_stats=pool_stats,
            training_bank=training_bank,
            matched_control_enabled=matched_control_enabled,
        )

        reference_distance = "hamming" if "hamming" in distances else distances[0]
        reference_mechanism = mechanisms[0]
        reference_epsilon = epsilon_values[0]
        reference_clip = clip_bounds[0]
        summary_parts = []
        for delta_value in delta_values:
            key = (
                "REPRESENTATIVE",
                float(delta_value),
                reference_distance,
                reference_mechanism,
                float(reference_epsilon),
                float(reference_clip),
            )
            if key not in aligned_stats:
                continue
            acc_mean, _ = aligned_stats[key].mean_std("attack_accuracy_percent")
            k_mean, _ = aligned_stats[key].mean_std("attack_hypothesis_count")
            rep_mean, _ = aligned_stats[key].mean_std("representative_count")
            summary_parts.append(
                f"d={float(delta_value):.2f}:K={k_mean:.2f},Rep={rep_mean:.2f},"
                f"Acc={acc_mean:.2f}%"
            )

        print(
            f"Run {run_index}/{pipeline_runs} finished | "
            + " | ".join(summary_parts)
        )
        print(
            f"saved={output_path} | expanded_templates={expanded_templates_this_run} | "
            f"run_elapsed={time.time() - run_start:.2f}s | "
            f"total_elapsed={time.time() - total_start:.2f}s"
        )

    print("\nAll pool-size-aligned inference results saved to:")
    print(output_path)
    print(f"Total elapsed: {time.time() - total_start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool-size-aligned black-box Sequence-DP user-requirement inference "
            "attack with representative-solution privacy evaluation."
        )
    )
    parser.add_argument("dataset", choices=["QWS", "Normal"])
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Processes used only for initial public hypothesis/template "
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
            "Optional pipeline-run override for a smoke test. Omit for "
            "dp_para['pipeline_runs']."
        ),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Build the minimum K=|Psi_opt| public attack bank and stop before "
            "Sequence-DP releases."
        ),
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Representative-pool max_utility_degradation values. Default: "
            "dp_para['delta_ablation_values'] when present, otherwise "
            "0.02 0.04 0.06 0.08 0.10."
        ),
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        choices=["PNF", "EM"],
        default=None,
        help="Optional mechanism subset. Omit for dp_para['noise_types'].",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of configured epsilon values.",
    )
    parser.add_argument(
        "--clip-bounds",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of configured clipping thresholds.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        choices=list(SUPPORTED_DISTANCES),
        default=None,
        help=(
            "Manuscript-permitted attack distances. Default: hamming jaccard qos."
        ),
    )
    parser.add_argument(
        "--no-matched-k-control",
        action="store_true",
        help=(
            "Disable the OPT_ONLY matched-hypothesis-count control. The main "
            "pool-aligned attack is unaffected."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_poolsize_representative_attack(
        dataset=args.dataset,
        workers=args.workers,
        task_limit=args.task_limit,
        rebuild_cache=args.rebuild_cache,
        validate_only=args.validate_only,
        mechanisms_filter=args.mechanisms,
        epsilons_filter=args.epsilons,
        clip_bounds_filter=args.clip_bounds,
        delta_values_filter=args.deltas,
        distances_filter=args.distances,
        runs_override=args.runs,
        matched_control_enabled=not args.no_matched_k_control,
    )


if __name__ == "__main__":
    main()
