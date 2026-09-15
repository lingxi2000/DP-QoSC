from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dp_perturb import Mechanisms, clip_scores, dp_para
from loadData_GA import SLATask, Service
from sla_core import EvalResult, SLAOnDemandGA, eval_key, evaluate_sequence


Chromosome = Tuple[int, ...]


@dataclass(frozen=True)
class SequenceCandidate:
    chromosome: Chromosome
    evaluation: EvalResult
    utility: float
    qos_vector: Tuple[float, float, float, float]


@dataclass
class CandidateSource:
    quality_candidates: List[SequenceCandidate]
    proxy_candidates: List[SequenceCandidate]
    proxy_vectors: np.ndarray
    proxy_similarity: np.ndarray


@dataclass
class CandidatePoolResult:
    candidates: List[SequenceCandidate]
    optimal_count: int
    representative_count: int
    proxy_count: int
    initial_representativeness: float
    representativeness: float
    expected_utility: float
    minimum_expected_utility: float


def candidate_utility(evaluation: EvalResult) -> float:
    q2_product = float(evaluation.q2_product)
    q3_product = float(evaluation.q3_product)
    utility = (q2_product + q3_product) / 2.0

    if utility < -1e-10 or utility > 1.0 + 1e-10:
        raise RuntimeError(
            "Sequence utility is outside [0, 1]. "
            f"Q2={q2_product}, Q3={q3_product}, U={utility}"
        )

    return min(1.0, max(0.0, utility))


def candidate_qos_vector(
    evaluation: EvalResult,
) -> Tuple[float, float, float, float]:
    values = (
        float(evaluation.q0_mean),
        float(evaluation.q1_min),
        float(evaluation.q2_product),
        float(evaluation.q3_product),
    )

    if any(value < -1e-10 or value > 1.0 + 1e-10 for value in values):
        raise RuntimeError(f"Aggregated QoS vector is outside [0, 1]: {values}")

    return tuple(min(1.0, max(0.0, value)) for value in values)


def make_candidate(
    chromosome: Chromosome,
    evaluation: EvalResult,
) -> SequenceCandidate:
    if evaluation.violations != 0:
        raise ValueError("Only globally feasible compositions may enter the pool.")

    return SequenceCandidate(
        chromosome=tuple(int(gene) for gene in chromosome),
        evaluation=evaluation,
        utility=candidate_utility(evaluation),
        qos_vector=candidate_qos_vector(evaluation),
    )


def decode_candidate(
    task: SLATask,
    candidate: SequenceCandidate,
) -> List[Service]:
    return [
        task.services[position][gene]
        for position, gene in enumerate(candidate.chromosome)
    ]


def _random_chromosome(
    category_sizes: Sequence[int],
    rng: random.Random,
) -> Chromosome:
    return tuple(rng.randrange(size) for size in category_sizes)


def _evaluate_chromosome(
    task: SLATask,
    chromosome: Chromosome,
) -> EvalResult:
    sequence = [
        task.services[position][gene]
        for position, gene in enumerate(chromosome)
    ]
    return evaluate_sequence(sequence, task.global_q2, task.global_q3)


def _candidate_sort_key(candidate: SequenceCandidate):
    return (
        eval_key(candidate.evaluation),
        candidate.chromosome,
    )


def _pairwise_similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("QoS matrices must be two-dimensional.")

    if left.shape[1] != right.shape[1]:
        raise ValueError("QoS matrices must use the same number of dimensions.")

    dimensions = left.shape[1]
    if dimensions <= 0:
        raise ValueError("QoS vector dimension must be positive.")

    distances = np.linalg.norm(
        left[:, None, :] - right[None, :, :],
        axis=2,
    )
    similarities = 1.0 - distances / math.sqrt(dimensions)
    return np.clip(similarities, 0.0, 1.0)


def build_candidate_source(
    task: SLATask,
    cfg,
    seed: int,
    optimal_count: int,
    proxy_pool_size: int,
) -> CandidateSource:

    optimal_count = int(optimal_count)
    proxy_pool_size = int(proxy_pool_size)

    if optimal_count <= 0:
        raise ValueError("optimal_count must be positive.")

    if proxy_pool_size <= 0:
        raise ValueError("proxy_pool_size must be positive.")

    model = SLAOnDemandGA(
        task,
        population_size=cfg.ga_population,
        max_generations=cfg.ga_max_generations,
        stagnation_patience=cfg.ga_stagnation_patience,
        crossover_rate=cfg.ga_crossover_rate,
        mutation_probability=cfg.ga_mutation_probability,
        elite_count=cfg.ga_elite_count,
        seed=seed,
    )
    model.search()

    ga_feasible: Dict[Chromosome, SequenceCandidate] = {}
    for chromosome, evaluation in model.cache.items():
        if evaluation.violations != 0:
            continue
        candidate = make_candidate(chromosome, evaluation)
        ga_feasible[candidate.chromosome] = candidate

    if len(ga_feasible) < optimal_count:
        raise RuntimeError(
            f"Task {task.task_index}: the baseline GA evaluated only "
            f"{len(ga_feasible)} unique globally feasible compositions, but "
            f"Psi_opt requires {optimal_count}. Increase the baseline search "
            "budget rather than filling Psi_opt with random proxy sequences."
        )

    quality_candidates = sorted(
        ga_feasible.values(),
        key=_candidate_sort_key,
    )[:optimal_count]
    quality_ids = {candidate.chromosome for candidate in quality_candidates}

    category_sizes = tuple(len(category) for category in task.services)
    rng = random.Random(seed ^ 0x5DEECE66D)
    proxy_by_id: Dict[Chromosome, SequenceCandidate] = {}
    max_attempts = max(5000, proxy_pool_size * 200)
    attempts = 0

    while len(proxy_by_id) < proxy_pool_size and attempts < max_attempts:
        chromosome = _random_chromosome(category_sizes, rng)
        attempts += 1

        if chromosome in quality_ids or chromosome in proxy_by_id:
            continue

        evaluation = _evaluate_chromosome(task, chromosome)
        if evaluation.violations != 0:
            continue

        proxy_by_id[chromosome] = make_candidate(chromosome, evaluation)

    if len(proxy_by_id) < proxy_pool_size:
        raise RuntimeError(
            f"Task {task.task_index}: independent rejection sampling found "
            f"only {len(proxy_by_id)} unique globally feasible proxy "
            f"compositions after {attempts} attempts, but V requires "
            f"{proxy_pool_size}. Increase the sampling budget or reduce the "
            "predeclared proxy_pool_size before formal evaluation."
        )

    proxy_candidates = list(proxy_by_id.values())
    proxy_vectors = np.asarray(
        [candidate.qos_vector for candidate in proxy_candidates],
        dtype=np.float64,
    )
    proxy_similarity = _pairwise_similarity(proxy_vectors, proxy_vectors)

    return CandidateSource(
        quality_candidates=quality_candidates,
        proxy_candidates=proxy_candidates,
        proxy_vectors=proxy_vectors,
        proxy_similarity=proxy_similarity,
    )


@lru_cache(maxsize=512)
def _unit_interval_gauss_legendre(order: int):
    if order <= 0:
        raise ValueError("Gauss-Legendre order must be positive.")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return nodes, weights


def _em_selection_probabilities(
    clipped_scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
) -> np.ndarray:
    logits = epsilon * clipped_scores / (2.0 * sensitivity)
    logits -= np.max(logits)
    weights = np.exp(logits)
    probabilities = weights / np.sum(weights)
    return probabilities.astype(np.float64, copy=False)


def _pnf_selection_probabilities(
    clipped_scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
) -> np.ndarray:

    empirical_max = float(np.max(clipped_scores))
    accept_probabilities = np.exp(
        epsilon * (clipped_scores - empirical_max) / (2.0 * sensitivity)
    )
    maximum_mask = np.isclose(
        clipped_scores,
        empirical_max,
        rtol=0.0,
        atol=1e-15,
    )
    accept_probabilities[maximum_mask] = 1.0
    accept_probabilities = np.clip(accept_probabilities, 0.0, 1.0)

    candidate_count = len(clipped_scores)
    quadrature_order = max(1, (candidate_count + 1) // 2)
    nodes, weights = _unit_interval_gauss_legendre(quadrature_order)

    factors = (
        (1.0 - accept_probabilities[:, None])
        + accept_probabilities[:, None] * nodes
    )
    product_all = np.prod(factors, axis=0)
    integrands = (
        accept_probabilities[:, None]
        * product_all[None, :]
        / factors
    )
    probabilities = np.sum(integrands * weights[None, :], axis=1)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    total_probability = float(np.sum(probabilities))
    if not math.isfinite(total_probability) or total_probability <= 0.0:
        raise RuntimeError("PNF output probabilities are invalid.")

    probabilities /= total_probability
    return probabilities.astype(np.float64, copy=False)


def selection_probabilities(
    candidates: Sequence[SequenceCandidate],
    mechanism: str,
    epsilon: Optional[float] = None,
    clip_bound: Optional[float] = None,
) -> np.ndarray:
    """Return the exact output PMF over a fixed candidate pool."""
    if not candidates:
        raise ValueError("Candidate pool cannot be empty.")

    mechanism = str(mechanism)
    raw_utilities = np.asarray(
        [candidate.utility for candidate in candidates],
        dtype=np.float64,
    )

    if mechanism == "Non_DP":
        best_utility = float(np.max(raw_utilities))
        tied_indices = [
            index
            for index, candidate in enumerate(candidates)
            if abs(candidate.utility - best_utility) <= 1e-15
        ]
        selected_index = min(
            tied_indices,
            key=lambda index: _candidate_sort_key(candidates[index]),
        )
        probabilities = np.zeros(len(candidates), dtype=np.float64)
        probabilities[selected_index] = 1.0
        return probabilities

    if mechanism not in {"PNF", "EM"}:
        raise ValueError(f"Unsupported selection mechanism: {mechanism}")

    if epsilon is None or clip_bound is None:
        raise ValueError("DP selection requires epsilon and clip_bound.")

    epsilon = float(epsilon)
    clip_bound = float(clip_bound)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if not 0.0 < clip_bound <= 1.0:
        raise ValueError("clip_bound must satisfy 0 < tau <= 1.")

    clipped_scores = np.minimum(raw_utilities, clip_bound)
    if mechanism == "EM":
        return _em_selection_probabilities(
            clipped_scores=clipped_scores,
            epsilon=epsilon,
            sensitivity=clip_bound,
        )

    return _pnf_selection_probabilities(
        clipped_scores=clipped_scores,
        epsilon=epsilon,
        sensitivity=clip_bound,
    )


def selection_probability_grid(
    candidates: Sequence[SequenceCandidate],
    mechanisms: Sequence[str],
    epsilon_values: Sequence[float],
    clip_bounds: Sequence[float],
) -> Dict[Tuple[str, float, float], np.ndarray]:

    if not candidates:
        raise ValueError("Candidate pool cannot be empty.")

    mechanism_values = [str(value) for value in mechanisms]
    unsupported = set(mechanism_values) - {"PNF", "EM"}
    if unsupported:
        raise ValueError(
            f"Unsupported selection mechanisms: {sorted(unsupported)}"
        )

    eps_array = np.asarray(
        [float(value) for value in epsilon_values],
        dtype=np.float64,
    )
    tau_values = [float(value) for value in clip_bounds]
    if eps_array.size == 0 or np.any(eps_array <= 0.0):
        raise ValueError("epsilon_values must contain positive values.")
    if not tau_values or any(not 0.0 < value <= 1.0 for value in tau_values):
        raise ValueError("Every clipping threshold must satisfy 0 < tau <= 1.")

    raw_utilities = np.asarray(
        [candidate.utility for candidate in candidates],
        dtype=np.float64,
    )
    candidate_count = len(raw_utilities)
    probabilities_by_setting: Dict[Tuple[str, float, float], np.ndarray] = {}

    needs_pnf = "PNF" in mechanism_values
    if needs_pnf:
        quadrature_order = max(1, (candidate_count + 1) // 2)
        nodes, quadrature_weights = _unit_interval_gauss_legendre(quadrature_order)

    for clip_bound in tau_values:
        clipped_scores = np.minimum(raw_utilities, clip_bound)

        if "EM" in mechanism_values:
            logits = (
                eps_array[:, None]
                * clipped_scores[None, :]
                / (2.0 * clip_bound)
            )
            logits -= np.max(logits, axis=1, keepdims=True)
            em_weights = np.exp(logits)
            em_probabilities = em_weights / np.sum(
                em_weights,
                axis=1,
                keepdims=True,
            )

            for epsilon_index, epsilon in enumerate(eps_array):
                probabilities_by_setting[("EM", float(epsilon), clip_bound)] = (
                    em_probabilities[epsilon_index].copy()
                )

        if needs_pnf:
            empirical_max = float(np.max(clipped_scores))
            accept_probabilities = np.exp(
                eps_array[:, None]
                * (clipped_scores[None, :] - empirical_max)
                / (2.0 * clip_bound)
            )
            maximum_mask = np.isclose(
                clipped_scores,
                empirical_max,
                rtol=0.0,
                atol=1e-15,
            )
            accept_probabilities[:, maximum_mask] = 1.0
            accept_probabilities = np.clip(accept_probabilities, 0.0, 1.0)

            factors = (
                (1.0 - accept_probabilities[:, :, None])
                + accept_probabilities[:, :, None] * nodes[None, None, :]
            )
            product_all = np.prod(factors, axis=1)
            integrands = (
                accept_probabilities[:, :, None]
                * product_all[:, None, :]
                / factors
            )
            pnf_probabilities = np.sum(
                integrands * quadrature_weights[None, None, :],
                axis=2,
            )
            pnf_probabilities = np.clip(pnf_probabilities, 0.0, 1.0)
            probability_totals = np.sum(
                pnf_probabilities,
                axis=1,
                keepdims=True,
            )
            if (
                not np.all(np.isfinite(probability_totals))
                or np.any(probability_totals <= 0.0)
            ):
                raise RuntimeError("PNF output probabilities are invalid.")
            pnf_probabilities /= probability_totals

            for epsilon_index, epsilon in enumerate(eps_array):
                probabilities_by_setting[("PNF", float(epsilon), clip_bound)] = (
                    pnf_probabilities[epsilon_index].copy()
                )

    return probabilities_by_setting


def expected_output_utility(
    candidates: Sequence[SequenceCandidate],
    mechanism: str,
    epsilon: float,
    clip_bound: float,
) -> float:

    probabilities = selection_probabilities(
        candidates=candidates,
        mechanism=mechanism,
        epsilon=epsilon,
        clip_bound=clip_bound,
    )
    raw_utilities = np.asarray(
        [candidate.utility for candidate in candidates],
        dtype=np.float64,
    )
    expected_utility = float(np.dot(probabilities, raw_utilities))

    if not math.isfinite(expected_utility):
        raise RuntimeError("Expected output utility is not finite.")

    return expected_utility

def _initial_proxy_coverage(
    source: CandidateSource,
    pool: Sequence[SequenceCandidate],
) -> np.ndarray:
    if not pool:
        return np.zeros(len(source.proxy_candidates), dtype=np.float64)

    pool_vectors = np.asarray(
        [candidate.qos_vector for candidate in pool],
        dtype=np.float64,
    )
    similarities = _pairwise_similarity(source.proxy_vectors, pool_vectors)
    return np.max(similarities, axis=1)


def build_representative_pool(
    source: CandidateSource,
    optimal_count: int,
    para=None,
) -> CandidatePoolResult:

    para = dp_para if para is None else para
    optimal_count = int(optimal_count)
    if optimal_count <= 0:
        raise ValueError("optimal_count must be positive.")

    if optimal_count > len(source.quality_candidates):
        raise ValueError(
            f"Requested {optimal_count} high-quality candidates, but source "
            f"contains only {len(source.quality_candidates)}."
        )

    rho = float(para.get("representative_rho", 0.8))
    delta = float(para.get("max_utility_degradation", 0.2))
    numerical_epsilon = float(
        para.get("representative_numerical_epsilon", 1e-12)
    )

    if not 0.0 < rho <= 1.0:
        raise ValueError("representative_rho must satisfy 0 < rho <= 1.")

    if not 0.0 <= delta <= 1.0:
        raise ValueError(
            "max_utility_degradation must satisfy 0 <= delta <= 1."
        )

    if numerical_epsilon <= 0.0:
        raise ValueError("representative_numerical_epsilon must be positive.")

    construction_mechanism = str(
        para.get("pool_construction_mechanism", "PNF")
    )
    construction_epsilon = float(
        para.get("pool_construction_epsilon", 1.0)
    )
    construction_clip_bound = float(
        para.get("pool_construction_clip_bound", 1.0)
    )

    if construction_mechanism not in {"PNF", "EM"}:
        raise ValueError(
            "pool_construction_mechanism must be 'PNF' or 'EM'."
        )

    if construction_epsilon <= 0.0:
        raise ValueError("pool_construction_epsilon must be positive.")

    if not 0.0 < construction_clip_bound <= 1.0:
        raise ValueError(
            "pool_construction_clip_bound must satisfy 0 < tau <= 1."
        )

    psi_opt = list(source.quality_candidates[:optimal_count])
    psi_opt_ids = {candidate.chromosome for candidate in psi_opt}

    remaining_indices = [
        index
        for index, candidate in enumerate(source.proxy_candidates)
        if candidate.chromosome not in psi_opt_ids
    ]

    current_pool = list(psi_opt)
    representative_candidates: List[SequenceCandidate] = []
    current_coverage = _initial_proxy_coverage(source, current_pool)
    initial_representativeness = float(np.mean(current_coverage))
    current_expected_utility = expected_output_utility(
        candidates=current_pool,
        mechanism=construction_mechanism,
        epsilon=construction_epsilon,
        clip_bound=construction_clip_bound,
    )
    reference_expected_utility = float(current_expected_utility)
    minimum_expected_utility = (
        1.0 - delta
    ) * reference_expected_utility

    while remaining_indices:
        remaining_array = np.asarray(remaining_indices, dtype=np.int64)
        similarities = source.proxy_similarity[:, remaining_array]
        gains = np.mean(
            np.maximum(similarities, current_coverage[:, None])
            - current_coverage[:, None],
            axis=0,
        )

        maximum_gain = float(np.max(gains))
        if maximum_gain <= 1e-15:
            break

        minimum_gain = rho * maximum_gain
        best_index = None
        best_ratio = -math.inf
        best_expected_utility = None
        best_chromosome = None

        for local_index, proxy_index in enumerate(remaining_indices):
            gain = float(gains[local_index])
            if gain + 1e-15 < minimum_gain:
                continue

            candidate = source.proxy_candidates[proxy_index]
            trial_pool = current_pool + [candidate]
            trial_expected_utility = expected_output_utility(
                candidates=trial_pool,
                mechanism=construction_mechanism,
                epsilon=construction_epsilon,
                clip_bound=construction_clip_bound,
            )

            if trial_expected_utility + 1e-15 < minimum_expected_utility:
                continue

            utility_loss = max(
                0.0,
                current_expected_utility - trial_expected_utility,
            )
            ratio = gain / (utility_loss + numerical_epsilon)

            if (
                ratio > best_ratio + 1e-15
                or (
                    abs(ratio - best_ratio) <= 1e-15
                    and (
                        best_chromosome is None
                        or candidate.chromosome < best_chromosome
                    )
                )
            ):
                best_index = proxy_index
                best_ratio = ratio
                best_expected_utility = trial_expected_utility
                best_chromosome = candidate.chromosome

        if best_index is None:
            break

        selected = source.proxy_candidates[best_index]
        current_pool.append(selected)
        representative_candidates.append(selected)
        current_expected_utility = float(best_expected_utility)

        selected_similarity = source.proxy_similarity[:, best_index]
        current_coverage = np.maximum(current_coverage, selected_similarity)
        remaining_indices.remove(best_index)

    representativeness = float(np.mean(current_coverage))

    return CandidatePoolResult(
        candidates=current_pool,
        optimal_count=len(psi_opt),
        representative_count=len(representative_candidates),
        proxy_count=len(source.proxy_candidates),
        initial_representativeness=initial_representativeness,
        representativeness=representativeness,
        expected_utility=current_expected_utility,
        minimum_expected_utility=minimum_expected_utility,
    )


def select_candidate(
    pool: CandidatePoolResult,
    mechanism: str,
    epsilon: Optional[float] = None,
    clip_bound: Optional[float] = None,
) -> SequenceCandidate:
    if not pool.candidates:
        raise ValueError("Candidate pool cannot be empty.")

    mechanism = str(mechanism)

    if mechanism == "Non_DP":
        best_utility = max(candidate.utility for candidate in pool.candidates)
        tied = [
            candidate
            for candidate in pool.candidates
            if abs(candidate.utility - best_utility) <= 1e-15
        ]
        return min(tied, key=_candidate_sort_key)

    if epsilon is None or clip_bound is None:
        raise ValueError("DP selection requires epsilon and clip_bound.")

    raw_scores = torch.tensor(
        [[candidate.utility for candidate in pool.candidates]],
        dtype=torch.float64,
    )
    clipped_scores = clip_scores(raw_scores, float(clip_bound))
    local_para = {
        "use_dp": True,
        "noise_type": mechanism,
        "epsilon": float(epsilon),
    }
    selected_index = int(
        Mechanisms.add_noise(
            scores=clipped_scores,
            dp_para_local=local_para,
            sensitivity=float(clip_bound),
        ).item()
    )
    return pool.candidates[selected_index]


def pool_mean_utility(pool: CandidatePoolResult) -> float:
    return float(
        np.mean([candidate.utility for candidate in pool.candidates])
    )
