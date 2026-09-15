from __future__ import annotations
import statistics
import sys
import time
from tqdm import tqdm
from config_loader import REFERENCE_FORMAT, load_config
from loadData_GA import load_sla_tasks, resolve_dataset_paths
from sla_core import exact_full_candidate, file_sha256, load_exact_reference, save_json_atomic

def parse_dataset():
    if len(sys.argv) != 2:
        print('Usage: python Generate_ExactReference.py <dataset>\nExample: python Generate_ExactReference.py QWS')
        raise SystemExit(2)
    return sys.argv[1]

def target_tasks_from_config(all_tasks, cfg):
    tasks = list(all_tasks)
    if cfg.exact_max_tasks is not None:
        tasks = tasks[:cfg.exact_max_tasks]
    return tasks

def task_ids(tasks):
    return [int(task.task_index) for task in tasks]

def new_reference(dataset, node_hash, service_hash, target_task_indices, cfg):
    return {'format': REFERENCE_FORMAT, 'meta': {'dataset': dataset, 'node_file': cfg.node_file, 'service_file': cfg.service_file, 'node_sha256': node_hash, 'service_sha256': service_hash, 'test_only': bool(cfg.test_only), 'exact_max_tasks': cfg.exact_max_tasks, 'target_task_count': len(target_task_indices), 'target_task_indices': list(target_task_indices), 'candidate_problem': 'FULL locally interval-feasible candidate problem', 'approximate_candidate_reduction': False, 'exact_dominance_pruning': True, 'local_candidate_constraint_semantics': 'interval', 'global_constraint_semantics': 'upper-only', 'exact_primary_objective': 'min |U2-Q2|+|U3-Q3| over globally feasible compositions', 'exact_secondary_tie_break': 'min (mean(q0)+1-min(q1))/2 only when primary distance is exactly equal', 'q0_aggregation': 'mean', 'q1_aggregation': 'min', 'q2_aggregation': 'product', 'q3_aggregation': 'product'}, 'tasks': []}

def validate_reference(reference, dataset, node_hash, service_hash, target_task_indices, cfg):
    meta = reference['meta']
    checks = [(reference.get('format') == REFERENCE_FORMAT, 'reference format'), (meta.get('dataset') == dataset, 'dataset'), (meta.get('node_file') == cfg.node_file, 'node_file'), (meta.get('service_file') == cfg.service_file, 'service_file'), (meta.get('node_sha256') == node_hash, 'node SHA-256'), (meta.get('service_sha256') == service_hash, 'service SHA-256'), (bool(meta.get('test_only')) == bool(cfg.test_only), 'test_only'), (meta.get('exact_max_tasks') == cfg.exact_max_tasks, 'EXACT.max_tasks'), (list(meta.get('target_task_indices', [])) == list(target_task_indices), 'target task set/order'), (int(meta.get('target_task_count', -1)) == len(target_task_indices), 'target task count'), (meta.get('approximate_candidate_reduction') is False, 'approximate candidate reduction'), (meta.get('exact_dominance_pruning') is True, 'Exact dominance pruning'), (meta.get('local_candidate_constraint_semantics') == 'interval', 'local candidate constraint semantics'), (meta.get('global_constraint_semantics') == 'upper-only', 'global constraint semantics'), (meta.get('exact_secondary_tie_break') is not None, 'Exact q0/q1 tie-break')]
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError(f'Existing Exact reference is incompatible with the current formal experiment.\nMismatched fields: {failed}\nDelete the old reference and regenerate the v5 reference.')
    stored_ids = [int(record['task_index']) for record in reference['tasks']]
    if len(stored_ids) != len(set(stored_ids)):
        raise RuntimeError('Existing Exact reference contains duplicate task_index records.')
    allowed = set(target_task_indices)
    unexpected = [task_index for task_index in stored_ids if task_index not in allowed]
    if unexpected:
        raise RuntimeError(f'Existing Exact reference contains tasks outside the current target set: {unexpected[:10]}')

def make_record(task, exact):
    return {'task_index': int(task.task_index), 'length': len(task.services), 'category_ids': [int(constraint.category_id) for constraint in task.local_constraints], 'candidate_counts': [len(category) for category in task.services], 'local_q2_intervals': [[float(constraint.q2[0]), float(constraint.q2[1])] for constraint in task.local_constraints], 'local_q3_intervals': [[float(constraint.q3[0]), float(constraint.q3[1])] for constraint in task.local_constraints], 'global_q2_upper': float(task.global_q2[1]), 'global_q3_upper': float(task.global_q3[1]), 'exact_distance': float(exact.distance), 'exact_performance': float(exact.performance_objective), 'exact_q0_mean': float(exact.q0_mean), 'exact_q1_min': float(exact.q1_min), 'exact_q2_product': float(exact.q2_product), 'exact_q3_product': float(exact.q3_product), 'exact_sequence_concrete_indices': [int(task.services[pos][gene][5]) for pos, gene in enumerate(exact.chromosome)], 'exact_method': exact.method, 'pareto_frontier_size': exact.frontier_size, 'exact_runtime': float(exact.runtime)}

def run(dataset):
    cfg = load_config()
    node_path, service_path = resolve_dataset_paths(dataset, cfg.node_file, cfg.service_file)
    node_hash = file_sha256(node_path)
    service_hash = file_sha256(service_path)
    all_tasks = load_sla_tasks(dataset, cfg.node_file, cfg.service_file, test_only=cfg.test_only, min_candidates=1)
    tasks = target_tasks_from_config(all_tasks, cfg)
    if not tasks:
        raise RuntimeError('No target task is available for Exact generation.')
    target_task_indices = task_ids(tasks)
    output_path = cfg.exact_reference_path(dataset)
    if cfg.exact_auto_resume and output_path.is_file():
        reference = load_exact_reference(output_path)
        validate_reference(reference, dataset, node_hash, service_hash, target_task_indices, cfg)
        completed = {int(record['task_index']) for record in reference['tasks']}
        print(f'[Auto resume] {len(completed)}/{len(tasks)} tasks already stored.')
    else:
        reference = new_reference(dataset, node_hash, service_hash, target_task_indices, cfg)
        completed = set()
    start_total = time.time()
    new_count = 0
    for task in tqdm(tasks, desc='Formal full-problem Exact v4'):
        if task.task_index in completed:
            continue
        exact = exact_full_candidate(task)
        if not exact.feasible:
            raise RuntimeError(f'Task {task.task_index} has no globally feasible composition under the current upper-bound semantics.')
        reference['tasks'].append(make_record(task, exact))
        completed.add(int(task.task_index))
        new_count += 1
        if new_count % cfg.exact_save_every == 0:
            reference['tasks'].sort(key=lambda x: int(x['task_index']))
            save_json_atomic(reference, output_path)
        if new_count % cfg.exact_progress_every == 0:
            print(f"[Progress] {len(reference['tasks'])}/{len(tasks)} | task={task.task_index} | D*={exact.distance:.10f} | F*={exact.performance_objective:.10f} | {exact.method}")
    reference['tasks'].sort(key=lambda x: int(x['task_index']))
    stored_ids = [int(record['task_index']) for record in reference['tasks']]
    if stored_ids != target_task_indices:
        raise RuntimeError(f'Exact generation finished with an incomplete or misordered reference.\nExpected {len(target_task_indices)} tasks, stored {len(stored_ids)}.')
    save_json_atomic(reference, output_path)
    distances = [float(record['exact_distance']) for record in reference['tasks']]
    performances = [float(record['exact_performance']) for record in reference['tasks']]
    runtimes = [float(record['exact_runtime']) for record in reference['tasks']]
    methods = {}
    for record in reference['tasks']:
        method = record['exact_method']
        methods[method] = methods.get(method, 0) + 1
    print('\n' + '#' * 116)
    print('FORMAL FULL-PROBLEM EXACT REFERENCE v4 FINISHED')
    print('#' * 116)
    print(f'Dataset                       : {dataset}')
    print(f'Node file                     : {cfg.node_file}')
    print(f'Test-only split               : {cfg.test_only}')
    print(f"Stored Exact tasks            : {len(reference['tasks'])}")
    print('Approx candidate reduction     : NO')
    print('Exact dominance pruning        : YES')
    print('Local candidate constraints    : L2<=q2<=U2, L3<=q3<=U3')
    print('Global constraints             : Q2<=U2, Q3<=U3')
    print('Exact secondary tie-break      : q0/q1 performance')
    print(f'Checkpoint every              : {cfg.exact_save_every} tasks')
    print(f'Exact methods                 : {methods}')
    print(f'Mean Exact D*                 : {statistics.mean(distances):.10f}')
    print(f'Median Exact D*               : {statistics.median(distances):.10f}')
    print(f'Mean Exact F*                 : {statistics.mean(performances):.10f}')
    print(f'Mean Exact runtime            : {statistics.mean(runtimes):.6f}s/task')
    print(f'Reference file                : {output_path}')
    print(f'Elapsed this invocation       : {time.time() - start_total:.2f}s')
    print('#' * 116)
if __name__ == '__main__':
    run(parse_dataset())