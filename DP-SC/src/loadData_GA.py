from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
Service = Tuple[float, float, float, float, int, int]
Interval = Tuple[float, float]

@dataclass(frozen=True)
class LocalConstraint:
    category_id: int
    q2: Interval
    q3: Interval
    raw_candidate_count: int
    feasible_candidate_count: int

@dataclass
class SLATask:
    task_index: int
    services: List[List[Service]]
    global_q2: Interval
    global_q3: Interval
    local_constraints: List[LocalConstraint]

def _normalize_interval(values: Sequence[float]) -> Interval:
    lower = float(values[0])
    upper = float(values[1])
    if lower > upper:
        lower, upper = (upper, lower)
    return (lower, upper)

def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    seen = set()
    for candidate in [here, *here.parents, cwd, *cwd.parents]:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / 'data').is_dir():
            return candidate
    raise FileNotFoundError('Cannot locate project root containing data/.')

def resolve_dataset_paths(dataset: str, node_filename: str, service_filename: str) -> Tuple[Path, Path]:
    root = find_project_root()
    dataset_dir = root / 'data' / str(dataset).strip()
    node_path = dataset_dir / node_filename
    service_path = dataset_dir / service_filename
    if not node_path.is_file():
        raise FileNotFoundError(node_path)
    if not service_path.is_file():
        raise FileNotFoundError(service_path)
    return (node_path, service_path)

def load_sla_tasks_from_files(node_path: str | Path, service_path: str | Path, test_only: bool=True, min_candidates: int=1) -> List[SLATask]:
    nodefeatures = json.loads(Path(node_path).read_text(encoding='utf-8'))
    service_feature = json.loads(Path(service_path).read_text(encoding='utf-8'))
    start = len(nodefeatures) * 3 // 4 if test_only else 0
    tasks: List[SLATask] = []
    for task_index, nodes in enumerate(nodefeatures[start:], start=start):
        global_node = None
        local_nodes = []
        for node in nodes:
            prefix = node[:-6]
            if 1 not in prefix:
                raise ValueError(f'Task {task_index}: invalid one-hot node.')
            category_id = prefix.index(1)
            if category_id == 0:
                global_node = node
            else:
                local_nodes.append((category_id, node))
        if global_node is None:
            raise ValueError(f'Task {task_index}: no global node.')
        global_q2 = _normalize_interval(global_node[-5:-3])
        global_q3 = _normalize_interval(global_node[-2:])
        task_services = []
        local_constraints = []
        for category_id, node in local_nodes:
            q2_interval = _normalize_interval(node[-5:-3])
            q3_interval = _normalize_interval(node[-2:])
            q2_lower, q2_upper = q2_interval
            q3_lower, q3_upper = q3_interval
            raw_services = service_feature[str(category_id)]
            candidates = []
            for concrete_index, raw_service in enumerate(raw_services):
                q0 = float(raw_service[-4])
                q1 = float(raw_service[-3])
                q2 = float(raw_service[-2])
                q3 = float(raw_service[-1])
                if q2_lower <= q2 <= q2_upper and q3_lower <= q3 <= q3_upper:
                    candidates.append((q0, q1, q2, q3, int(category_id), int(concrete_index)))
            if len(candidates) < min_candidates:
                raise RuntimeError(f'Task {task_index}, category {category_id}: {len(candidates)} interval-feasible candidates.')
            task_services.append(candidates)
            local_constraints.append(LocalConstraint(category_id=int(category_id), q2=q2_interval, q3=q3_interval, raw_candidate_count=len(raw_services), feasible_candidate_count=len(candidates)))
        tasks.append(SLATask(task_index=int(task_index), services=task_services, global_q2=global_q2, global_q3=global_q3, local_constraints=local_constraints))
    return tasks

def load_sla_tasks(dataset: str, node_filename: str, service_filename: str, test_only: bool=True, min_candidates: int=1) -> List[SLATask]:
    node_path, service_path = resolve_dataset_paths(dataset, node_filename, service_filename)
    print('\n========== Independent SLA Data Loader ==========')
    print(f'dataset              : {dataset}')
    print(f'nodefeatures         : {node_path}')
    print(f'serviceFeature       : {service_path}')
    print(f'test_only            : {test_only}')
    print('local q2/q3 filter   : full interval [lower, upper]')
    print('global q2/q3         : upper-bound constraints')
    print('GNN/PN files         : not used')
    print('=================================================\n')
    return load_sla_tasks_from_files(node_path, service_path, test_only=test_only, min_candidates=min_candidates)