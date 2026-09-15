from __future__ import annotations
import hashlib
import json
import itertools
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch
from loadData_GA import Service, SLATask
TOL = 1e-12
BOUND_TOL = 1e-15
_GA_BACKEND_ANNOUNCED = False

@dataclass(frozen=True)
class EvalResult:
    violations: int
    violation_amount: float
    distance: float
    performance_objective: float
    d_q2: float
    d_q3: float
    q0_mean: float
    q1_min: float
    q2_product: float
    q3_product: float

@dataclass(frozen=True)
class ExactResult:
    feasible: bool
    distance: Optional[float]
    performance_objective: Optional[float]
    q0_mean: Optional[float]
    q1_min: Optional[float]
    q2_product: Optional[float]
    q3_product: Optional[float]
    chromosome: Optional[Tuple[int, ...]]
    method: str
    runtime: float
    frontier_size: Optional[int]

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def evaluate_sequence(sequence: Sequence[Service], global_q2, global_q3) -> EvalResult:
    if not sequence:
        raise ValueError('Empty composition.')
    q0 = np.asarray([float(s[0]) for s in sequence], dtype=np.float64)
    q1 = np.asarray([float(s[1]) for s in sequence], dtype=np.float64)
    q2 = np.asarray([float(s[2]) for s in sequence], dtype=np.float64)
    q3 = np.asarray([float(s[3]) for s in sequence], dtype=np.float64)
    q0_mean = float(np.mean(q0))
    q1_min = float(np.min(q1))
    q2_product = float(np.prod(q2))
    q3_product = float(np.prod(q3))
    u2 = float(global_q2[1])
    u3 = float(global_q3[1])
    amount2 = max(0.0, q2_product - u2)
    amount3 = max(0.0, q3_product - u3)
    v2 = int(amount2 > TOL)
    v3 = int(amount3 > TOL)
    d_q2 = abs(u2 - q2_product)
    d_q3 = abs(u3 - q3_product)
    performance = float((q0_mean + 1.0 - q1_min) / 2.0)
    return EvalResult(violations=v2 + v3, violation_amount=amount2 + amount3, distance=d_q2 + d_q3, performance_objective=performance, d_q2=d_q2, d_q3=d_q3, q0_mean=q0_mean, q1_min=q1_min, q2_product=q2_product, q3_product=q3_product)

def eval_key(result: EvalResult):

    return (result.violations, result.violation_amount, result.distance, result.performance_objective)

class SLAOnDemandGA:
    def __init__(
        self,
        task: SLATask,
        population_size=80,
        max_generations=120,
        stagnation_patience=20,
        crossover_rate=1.0,
        mutation_probability=1.0,
        elite_count=4,
        seed=None,
    ):
        self.task = task
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.stagnation_patience = int(stagnation_patience)
        self.crossover_rate = float(crossover_rate)
        self.mutation_probability = float(mutation_probability)
        self.elite_count = min(
            max(1, int(elite_count)),
            self.population_size - 1,
        )
        self.rng = random.Random(seed)
        self.length = len(task.services)
        self.cache: Dict[Tuple[int, ...], EvalResult] = {}
        self._sort_key_cache = {}
        self.generations_used = 0

        if self.length == 0:
            raise ValueError("Task has no abstract-service position.")

        if self.population_size < 4:
            raise ValueError("population_size must be >= 4.")

        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive.")

        if self.stagnation_patience <= 0:
            raise ValueError("stagnation_patience must be positive.")

        if self.stagnation_patience > self.max_generations:
            raise ValueError(
                "stagnation_patience cannot exceed max_generations."
            )

        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1].")

        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError(
                "mutation_probability must be in [0, 1]."
            )

        for pos, category in enumerate(task.services):
            if not category:
                raise ValueError(
                    f"Empty candidate set at position {pos}."
                )

        self._category_sizes = tuple(
            len(category)
            for category in self.task.services
        )
        self._mutable_positions = tuple(
            pos
            for pos, size in enumerate(self._category_sizes)
            if size > 1
        )

        roulette_weights = range(
            self.population_size,
            0,
            -1,
        )
        self._roulette_cum_weights = tuple(
            itertools.accumulate(roulette_weights)
        )

        self._fitness_device = self._resolve_fitness_device()
        self._gpu_service_qos = None
        self._gpu_positions = None
        self._gpu_u2 = None
        self._gpu_u3 = None

        if self._fitness_device is not None:
            self._prepare_gpu_fitness_data()

        self._announce_fitness_backend_once()

    @staticmethod
    def _resolve_fitness_device():
        requested = os.environ.get("SLA_GA_DEVICE", "auto").strip().lower()

        if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
            raise ValueError(
                "SLA_GA_DEVICE must be one of: auto, cpu, cuda, cuda:<index>."
            )

        if requested == "cpu":
            return None

        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return None

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"SLA_GA_DEVICE={requested!r}, but CUDA is unavailable in the "
                "current PyTorch installation/runtime."
            )

        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device {device.index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return device

    def _prepare_gpu_fitness_data(self):
        max_candidates = max(self._category_sizes)
        qos = np.zeros(
            (self.length, max_candidates, 4),
            dtype=np.float64,
        )

        for position, category in enumerate(self.task.services):
            for gene, service in enumerate(category):
                qos[position, gene, 0] = float(service[0])
                qos[position, gene, 1] = float(service[1])
                qos[position, gene, 2] = float(service[2])
                qos[position, gene, 3] = float(service[3])

        self._gpu_service_qos = torch.as_tensor(
            qos,
            dtype=torch.float64,
            device=self._fitness_device,
        )
        self._gpu_positions = torch.arange(
            self.length,
            dtype=torch.long,
            device=self._fitness_device,
        ).unsqueeze(0)
        self._gpu_u2 = torch.tensor(
            float(self.task.global_q2[1]),
            dtype=torch.float64,
            device=self._fitness_device,
        )
        self._gpu_u3 = torch.tensor(
            float(self.task.global_q3[1]),
            dtype=torch.float64,
            device=self._fitness_device,
        )

    def _announce_fitness_backend_once(self):
        global _GA_BACKEND_ANNOUNCED
        if _GA_BACKEND_ANNOUNCED:
            return

        if self._fitness_device is None:
            print(
                "[SLA-GA] fitness backend: CPU (original per-sequence evaluator)",
                flush=True,
            )
        else:
            device_name = torch.cuda.get_device_name(self._fitness_device)
            print(
                "[SLA-GA] fitness backend: CUDA batch evaluator | "
                f"device={self._fitness_device} | name={device_name} | dtype=float64",
                flush=True,
            )
        _GA_BACKEND_ANNOUNCED = True

    def _random(self):
        return tuple(
            self.rng.randrange(size)
            for size in self._category_sizes
        )

    def _decode(self, chromosome):
        return [
            self.task.services[pos][gene]
            for pos, gene in enumerate(chromosome)
        ]

    def _evaluate(self, chromosome):
        if chromosome not in self.cache:

            self.cache[chromosome] = evaluate_sequence(
                self._decode(chromosome),
                self.task.global_q2,
                self.task.global_q3,
            )

        return self.cache[chromosome]

    def _evaluate_population(self, population):
        missing = []
        seen = set()

        for chromosome in population:
            if chromosome in self.cache or chromosome in seen:
                continue
            seen.add(chromosome)
            missing.append(chromosome)

        if not missing:
            return

        if self._fitness_device is None:
            for chromosome in missing:
                self.cache[chromosome] = evaluate_sequence(
                    self._decode(chromosome),
                    self.task.global_q2,
                    self.task.global_q3,
                )
            return

        genes = torch.tensor(
            missing,
            dtype=torch.long,
            device=self._fitness_device,
        )
        positions = self._gpu_positions.expand(genes.shape[0], -1)

        with torch.inference_mode():
            selected_qos = self._gpu_service_qos[positions, genes]

            q0_mean = torch.mean(selected_qos[:, :, 0], dim=1)
            q1_min = torch.amin(selected_qos[:, :, 1], dim=1)
            q2_product = torch.prod(selected_qos[:, :, 2], dim=1)
            q3_product = torch.prod(selected_qos[:, :, 3], dim=1)

            amount2 = torch.clamp(q2_product - self._gpu_u2, min=0.0)
            amount3 = torch.clamp(q3_product - self._gpu_u3, min=0.0)
            violations = (
                (amount2 > TOL).to(torch.float64)
                + (amount3 > TOL).to(torch.float64)
            )

            d_q2 = torch.abs(self._gpu_u2 - q2_product)
            d_q3 = torch.abs(self._gpu_u3 - q3_product)
            distance = d_q2 + d_q3
            violation_amount = amount2 + amount3
            performance = (q0_mean + 1.0 - q1_min) / 2.0

            packed = torch.stack(
                (
                    violations,
                    violation_amount,
                    distance,
                    performance,
                    d_q2,
                    d_q3,
                    q0_mean,
                    q1_min,
                    q2_product,
                    q3_product,
                ),
                dim=1,
            ).cpu().numpy()

        for chromosome, values in zip(missing, packed):
            self.cache[chromosome] = EvalResult(
                violations=int(values[0]),
                violation_amount=float(values[1]),
                distance=float(values[2]),
                performance_objective=float(values[3]),
                d_q2=float(values[4]),
                d_q3=float(values[5]),
                q0_mean=float(values[6]),
                q1_min=float(values[7]),
                q2_product=float(values[8]),
                q3_product=float(values[9]),
            )

    def _sort_key(self, chromosome):
        key = self._sort_key_cache.get(chromosome)

        if key is None:
            key = (
                eval_key(self._evaluate(chromosome)),
                chromosome,
            )
            self._sort_key_cache[chromosome] = key

        return key

    def _roulette(self, population):
        return self.rng.choices(
            population,
            cum_weights=self._roulette_cum_weights,
            k=1,
        )[0]

    def _crossover(self, parent_a, parent_b):

        if (
            self.length <= 1
            or self.rng.random() > self.crossover_rate
        ):
            return parent_a, parent_b

        cut = self.rng.randrange(1, self.length)

        child_a = parent_a[:cut] + parent_b[cut:]
        child_b = parent_b[:cut] + parent_a[cut:]

        return child_a, child_b

    def _mutate(self, chromosome):
        """Single-gene mutation with the original random-number call order."""
        if self.rng.random() > self.mutation_probability:
            return chromosome

        if not self._mutable_positions:
            return chromosome

        position = self.rng.choice(self._mutable_positions)
        genes = list(chromosome)
        current = genes[position]
        category_size = self._category_sizes[position]

        new_gene = self.rng.randrange(category_size - 1)

        if new_gene >= current:
            new_gene += 1

        genes[position] = new_gene
        return tuple(genes)

    def search(self):

        population = [
            self._random()
            for _ in range(self.population_size)
        ]

        self._evaluate_population(population)
        population.sort(key=self._sort_key)
        best = population[0]
        best_objective = eval_key(self._evaluate(best))
        stagnation = 0
        self.generations_used = 0

        for generation in range(1, self.max_generations + 1):
            next_population = population[: self.elite_count].copy()

            while len(next_population) < self.population_size:
                parent_a = self._roulette(population)
                parent_b = self._roulette(population)

                child_a, child_b = self._crossover(
                    parent_a,
                    parent_b,
                )

                next_population.append(
                    self._mutate(child_a)
                )

                if len(next_population) < self.population_size:
                    next_population.append(
                        self._mutate(child_b)
                    )

            population = next_population
            self._evaluate_population(population)
            population.sort(key=self._sort_key)

            current = population[0]
            current_objective = eval_key(
                self._evaluate(current)
            )

            if current_objective < best_objective:
                best = current
                best_objective = current_objective
                stagnation = 0
            else:
                if (
                    current_objective == best_objective
                    and current < best
                ):
                    best = current

                stagnation += 1

            self.generations_used = generation

            if stagnation >= self.stagnation_patience:
                break

        return self._decode(best), self._evaluate(best)

def _all_compositions_upper_feasible(task):

    max_q2_product = 1.0
    max_q3_product = 1.0
    for category in task.services:
        max_q2_product *= max((float(s[2]) for s in category))
        max_q3_product *= max((float(s[3]) for s in category))
    return max_q2_product <= float(task.global_q2[1]) and max_q3_product <= float(task.global_q3[1])

def _positive_q2_q3(task):

    return all((float(service[2]) > 0.0 and float(service[3]) > 0.0 for category in task.services for service in category))

def _secondary_dominates(q0_left, q1_left, q0_right, q1_right):

    return q0_left <= q0_right and q1_left >= q1_right and (q0_left < q0_right or q1_left > q1_right)

def _prune_equal_primary_candidates(candidates):

    kept = []
    for candidate in candidates:
        dominated = False
        survivors = []
        for old in kept:
            if _secondary_dominates(old[2], old[3], candidate[2], candidate[3]):
                dominated = True
                break
            if not _secondary_dominates(candidate[2], candidate[3], old[2], old[3]):
                survivors.append(old)
        if not dominated:
            survivors.append(candidate)
            kept = survivors
    return kept

def _pareto_prune_candidates_exact(category):

    indexed = [(float(service[2]), float(service[3]), float(service[0]), float(service[1]), int(gene)) for gene, service in enumerate(category)]
    indexed.sort(key=lambda x: (-x[0], -x[1], x[2], -x[3], x[4]))
    kept = []
    best_q3_from_higher_q2 = -math.inf
    i = 0
    n = len(indexed)
    while i < n:
        q2_value = indexed[i][0]
        j = i
        while j < n and indexed[j][0] == q2_value:
            j += 1
        q2_group = indexed[i:j]
        max_q3 = q2_group[0][1]
        max_q3_candidates = [candidate for candidate in q2_group if candidate[1] == max_q3]
        if best_q3_from_higher_q2 < max_q3:
            kept.extend(_prune_equal_primary_candidates(max_q3_candidates))
            best_q3_from_higher_q2 = max(best_q3_from_higher_q2, max_q3)
        i = j
    return kept

def _tie_state_dominates(left, right):

    return _secondary_dominates(left[2], left[3], right[2], right[3])

def _prune_equal_primary_states(states):

    kept = []
    for state in states:
        dominated = False
        survivors = []
        for old in kept:
            if _tie_state_dominates(old, state):
                dominated = True
                break
            if not _tie_state_dominates(state, old):
                survivors.append(old)
        if not dominated:
            survivors.append(state)
            kept = survivors
    return kept

def _pareto_prune_states_exact(states):
    if not states:
        return []
    states.sort(key=lambda x: (-x[0], -x[1], x[4]))
    kept = []
    best_q3_from_higher_p2 = -math.inf
    i = 0
    n = len(states)
    while i < n:
        p2_value = states[i][0]
        j = i
        while j < n and states[j][0] == p2_value:
            j += 1
        p2_group = states[i:j]
        max_q3 = p2_group[0][1]
        max_q3_states = [state for state in p2_group if state[1] == max_q3]
        if best_q3_from_higher_p2 < max_q3:
            kept.extend(_prune_equal_primary_states(max_q3_states))
            best_q3_from_higher_p2 = max(best_q3_from_higher_p2, max_q3)
        i = j
    return kept

def _exact_pareto_dp(task):

    candidate_frontiers = [_pareto_prune_candidates_exact(category) for category in task.services]
    frontier = [(1.0, 1.0, 0.0, math.inf, tuple())]
    for candidate_frontier in candidate_frontiers:
        combined = []
        for p2, p3, q0_sum, q1_min, chromosome in frontier:
            for q2, q3, q0, q1, gene in candidate_frontier:
                combined.append((p2 * q2, p3 * q3, q0_sum + q0, min(q1_min, q1), chromosome + (gene,)))
        frontier = _pareto_prune_states_exact(combined)
    if not frontier:
        return ExactResult(feasible=False, distance=None, performance_objective=None, q0_mean=None, q1_min=None, q2_product=None, q3_product=None, chromosome=None, method='full-candidate exact Pareto-DP-v5', runtime=0.0, frontier_size=0)
    best_chromosome = None
    best_eval = None
    for state in frontier:
        chromosome = state[4]
        sequence = [task.services[pos][gene] for pos, gene in enumerate(chromosome)]
        ev = evaluate_sequence(sequence, task.global_q2, task.global_q3)
        key = (eval_key(ev), chromosome)
        if best_eval is None or key < (eval_key(best_eval), best_chromosome):
            best_chromosome = chromosome
            best_eval = ev
    return ExactResult(feasible=best_eval.violations == 0, distance=float(best_eval.distance), performance_objective=float(best_eval.performance_objective), q0_mean=float(best_eval.q0_mean), q1_min=float(best_eval.q1_min), q2_product=float(best_eval.q2_product), q3_product=float(best_eval.q3_product), chromosome=tuple(best_chromosome), method='full-candidate exact Pareto-DP-v5', runtime=0.0, frontier_size=len(frontier))

def _exact_branch_and_bound(task):

    n = len(task.services)
    u2 = float(task.global_q2[1])
    u3 = float(task.global_q3[1])
    candidates = []
    for category in task.services:
        ordered = sorted(enumerate(category), key=lambda x: (-(float(x[1][2]) + float(x[1][3])), float(x[1][0]), -float(x[1][1]), x[0]))
        candidates.append(ordered)
    max_rem2 = [1.0] * (n + 1)
    max_rem3 = [1.0] * (n + 1)
    min_rem2 = [1.0] * (n + 1)
    min_rem3 = [1.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        category = task.services[i]
        max_rem2[i] = max_rem2[i + 1] * max((float(s[2]) for s in category))
        max_rem3[i] = max_rem3[i + 1] * max((float(s[3]) for s in category))
        min_rem2[i] = min_rem2[i + 1] * min((float(s[2]) for s in category))
        min_rem3[i] = min_rem3[i + 1] * min((float(s[3]) for s in category))
    best_distance = math.inf
    best_perf = math.inf
    best_chromosome = None
    genes = [0] * n

    def dfs(depth, p2, p3, q0_sum, q1_min):
        nonlocal best_distance
        nonlocal best_perf
        nonlocal best_chromosome
        if p2 * min_rem2[depth] > u2 + TOL or p3 * min_rem3[depth] > u3 + TOL:
            return
        optimistic_q2 = min(u2, p2 * max_rem2[depth])
        optimistic_q3 = min(u3, p3 * max_rem3[depth])
        distance_lower_bound = abs(u2 - optimistic_q2) + abs(u3 - optimistic_q3)
        if distance_lower_bound > best_distance + BOUND_TOL:
            return
        if depth == n:
            if p2 <= u2 + TOL and p3 <= u3 + TOL:
                distance = abs(u2 - p2) + abs(u3 - p3)
                q0_mean = q0_sum / n
                perf = (q0_mean + 1.0 - q1_min) / 2.0
                if distance < best_distance:
                    best_distance = distance
                    best_perf = perf
                    best_chromosome = tuple(genes)
                elif distance == best_distance and perf < best_perf:
                    best_perf = perf
                    best_chromosome = tuple(genes)
            return
        for gene, service in candidates[depth]:
            genes[depth] = gene
            dfs(depth + 1, p2 * float(service[2]), p3 * float(service[3]), q0_sum + float(service[0]), min(q1_min, float(service[1])))
    dfs(0, 1.0, 1.0, 0.0, math.inf)
    if best_chromosome is None:
        return ExactResult(feasible=False, distance=None, performance_objective=None, q0_mean=None, q1_min=None, q2_product=None, q3_product=None, chromosome=None, method='full-candidate exact branch-and-bound-v5', runtime=0.0, frontier_size=None)
    sequence = [task.services[pos][gene] for pos, gene in enumerate(best_chromosome)]
    ev = evaluate_sequence(sequence, task.global_q2, task.global_q3)
    if ev.violations != 0:
        raise RuntimeError('Internal Exact error: branch-and-bound selected a sequence that evaluate_sequence marks infeasible.')
    return ExactResult(feasible=True, distance=float(ev.distance), performance_objective=float(ev.performance_objective), q0_mean=float(ev.q0_mean), q1_min=float(ev.q1_min), q2_product=float(ev.q2_product), q3_product=float(ev.q3_product), chromosome=tuple(best_chromosome), method='full-candidate exact branch-and-bound-v5', runtime=0.0, frontier_size=None)

def exact_full_candidate(task):
    start = time.time()
    if _all_compositions_upper_feasible(task) and _positive_q2_q3(task):
        result = _exact_pareto_dp(task)
    else:
        result = _exact_branch_and_bound(task)
    return ExactResult(feasible=result.feasible, distance=result.distance, performance_objective=result.performance_objective, q0_mean=result.q0_mean, q1_min=result.q1_min, q2_product=result.q2_product, q3_product=result.q3_product, chromosome=result.chromosome, method=result.method, runtime=time.time() - start, frontier_size=result.frontier_size)

def boundary_optimality(exact_distance, solution_distance, tolerance=1e-10):

    exact_distance = float(exact_distance)
    solution_distance = float(solution_distance)
    if solution_distance < exact_distance - tolerance:
        raise RuntimeError(f'A solution distance is smaller than Exact D*. The Exact reference, data, or objective is inconsistent.\nD*={exact_distance}, D_solution={solution_distance}')
    if abs(solution_distance - exact_distance) <= tolerance:
        return 1.0
    if exact_distance <= 1e-15:
        return 0.0
    return exact_distance / solution_distance

def distance_gap(exact_distance, solution_distance, tolerance=1e-10):
    gap = float(solution_distance) - float(exact_distance)
    if gap < -tolerance:
        raise RuntimeError('Negative gap: a solution appears better than Exact.')
    return max(0.0, gap)

def save_json_atomic(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(output_path)

def load_exact_reference(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('format') != 'SLA-FullExactBoundary-v5':
        raise ValueError('Unsupported / obsolete Exact reference. Generate a new reference.')
    return data
