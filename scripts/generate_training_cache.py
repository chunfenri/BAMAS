import torch
import random
import os
import json
import argparse
import sys
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import queue
import threading
import time
sys.path.insert(0, os.getcwd())
from eapae_agent_sys.utils.config_loader import ConfigLoader
from eapae_agent_sys.data_processing.gsm8k_loader import load_gsm8k_dataset
from eapae_agent_sys.planning.semantic_filter import SemanticFilter
from eapae_agent_sys.planning.low_level_instantiator import LowLevelInstantiator
from eapae_agent_sys.execution.execution_engine import ExecutionEngine
from eapae_agent_sys.utils.llm_api import LLM_API
from eapae_agent_sys.utils.evaluation import evaluate_success
def monitor_workers(workers: list, heartbeat_dict: dict, stop_event: threading.Event, timeout: int):
    """
    A thread that monitors worker processes and terminates them if they hang.
    It checks a shared dictionary for the last heartbeat from each worker.
    """
    while not stop_event.wait(20.0):
        for p in workers:
            if not p.is_alive():
                continue
            last_beat = heartbeat_dict.get(p.pid)
            if last_beat is None:
                continue
            if time.time() - last_beat > timeout:
                p.terminate()
def worker(work_queue, result_queue, heartbeat_dict, incorrect_queue):
    """Worker process to generate a single experience."""
    worker_pid = os.getpid()
    heartbeat_dict[worker_pid] = time.time()
    config_loader = ConfigLoader()
    llm_api = LLM_API()
    semantic_filter = SemanticFilter(config_loader, llm_api)
    low_level_instantiator = LowLevelInstantiator(config_loader)
    execution_engine = ExecutionEngine(config_loader, low_level_instantiator)
    while True:
        work_item = work_queue.get()
        if work_item is None:
            break
        heartbeat_dict[worker_pid] = time.time()
        work_id, task, budget, pattern, pattern_idx, behavior_log_prob = work_item
        worker_pid = os.getpid()
        candidate_agents = semantic_filter.filter_candidates(task['question'], collaboration_pattern=pattern)
        is_feasible = False
        is_correct = False
        actual_cost = 0.0
        final_context = None
        if not candidate_agents:
            final_context = {"history": ["Planning failed: Semantic filter returned no candidate agents."]}
        else:
            agent_pool, is_feasible = low_level_instantiator.solve(
                pattern_or_name=pattern,
                budget=budget,
                candidate_agent_ids=candidate_agents
            )
            if is_feasible:
                final_context, actual_cost = execution_engine.execute_hybrid(
                    agent_pool, budget, task['question'], collaboration_pattern=pattern
                )
                success_code = evaluate_success(final_context, task['answer'])
                is_correct = success_code == 1
            else:
                final_context = {"history": ["Planning failed: No feasible agent team could be formed for the given budget and pattern."]}
        experience = {
            "work_id": work_id,
            "task_description": task['question'], 
            "budget": budget,
            "action": pattern_idx, 
            "is_correct": is_correct,
            "actual_cost": actual_cost, 
            "planning_feasible": is_feasible,
            "behavior_log_prob": behavior_log_prob
        }
        result_queue.put(experience)
        if not is_correct:
            incorrect_experience = {
                "work_id": work_id,
                "task_description": task['question'],
                "ground_truth_answer": task['answer'], 
                "budget": budget,
                "action": pattern_idx,
                "pattern_name": pattern['name'],
                "is_correct": is_correct,
                "actual_cost": actual_cost,
                "planning_feasible": is_feasible,
                "behavior_log_prob": behavior_log_prob,
                "execution_history": final_context.get('history', []) if final_context else [],
                "full_context": final_context
            }
            incorrect_queue.put(incorrect_experience)
def generate_cache_single_pattern_budget(args, pattern_idx: int, budget_idx: int):
    """
    Generate specifiedpatternandbudgetcache data，output format matchesmainfunction completely consistent
    Args:
        args: Command line arguments
        pattern_idx: patternindex (0=linear, 1=star, 2=feedback, 3=planner_driven)
        budget_idx: budgetindex (0-4correspondingnp.linspaceof5values)
    """
    print(f"Target: pattern_idx={pattern_idx}, budget_idx={budget_idx}")
    mp.set_start_method("spawn", force=True)
    config_loader = ConfigLoader()
    params = config_loader.params
    train_dataset = load_gsm8k_dataset(path=params['data']['gsm8k_train_path'])
    random.seed(params['training']['seed'])
    random.shuffle(train_dataset)
    total_available = len(train_dataset)
    start_idx = args.begin_with
    if start_idx >= total_available:
        raise ValueError(f"begin_with ({start_idx}) exceeds dataset size ({total_available})")
    if args.num_samples == -1:
        end_idx = total_available
    else:
        end_idx = min(start_idx + args.num_samples, total_available)
    train_dataset = train_dataset[start_idx:end_idx]
    actual_samples = len(train_dataset)
    all_patterns = config_loader.patterns['patterns']
    num_patterns = len(all_patterns)
    if pattern_idx >= num_patterns:
        raise ValueError(f"pattern_idx ({pattern_idx}) exceeds available patterns ({num_patterns})")
    target_pattern = all_patterns[pattern_idx]
    behavior_log_prob = np.log(1.0 / num_patterns)
    budgets_to_test = np.linspace(*params['training']['budget_range'], num=args.num_budget_steps).tolist()
    if budget_idx >= len(budgets_to_test):
        raise ValueError(f"budget_idx ({budget_idx}) exceeds available budgets ({len(budgets_to_test)})")
    target_budget = budgets_to_test[budget_idx]
    print(f"Target budget: {target_budget}")
    manager = mp.Manager()
    heartbeat_dict = manager.dict()
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    incorrect_queue = mp.Queue()
    work_items = []
    work_items_map = {}
    for i, task in enumerate(train_dataset):
        global_task_idx = start_idx + i
        work_id = f"task_{global_task_idx}_budget_{budget_idx}_pattern_{pattern_idx}"
        work_item = (work_id, task, target_budget, target_pattern, pattern_idx, behavior_log_prob)
        work_items.append(work_item)
        work_items_map[work_id] = work_item
    for item in work_items:
        work_queue.put(item)
    for _ in range(args.num_workers):
        work_queue.put(None)
    workers = []
    for _ in range(args.num_workers):
        p = mp.Process(target=worker, args=(work_queue, result_queue, heartbeat_dict, incorrect_queue))
        p.start()
        workers.append(p)
    TIMEOUT_SECONDS = 300
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_workers,
        args=(workers, heartbeat_dict, stop_event, TIMEOUT_SECONDS),
        daemon=True
    )
    monitor_thread.start()
    offline_dataset = []
    incorrect_res_data = []
    total_work_items = len(work_items)
    try:
        with tqdm(total=total_work_items, desc=f"Generating Pattern-{pattern_idx} Budget-{budget_idx}") as pbar:
            while pbar.n < total_work_items:
                try:
                    experience = result_queue.get(timeout=5)
                    offline_dataset.append(experience)
                    pbar.update(1)
                    try:
                        while True:
                            incorrect_exp = incorrect_queue.get_nowait()
                            incorrect_res_data.append(incorrect_exp)
                    except queue.Empty:
                        pass
                except queue.Empty:
                    if not any(p.is_alive() for p in workers):
                        break
    finally:
        print("\n--- Initiating Cleanup ---")
        try:
            while True:
                incorrect_exp = incorrect_queue.get_nowait()
                incorrect_res_data.append(incorrect_exp)
        except queue.Empty:
            pass
        successful_ids = {exp.get('work_id') for exp in offline_dataset if exp.get('work_id')}
        all_ids = set(work_items_map.keys())
        failed_ids = all_ids - successful_ids
        if failed_ids:
            print(f"[Cleanup] Found {len(failed_ids)} incomplete tasks. Adding them as negative experiences.")
            for failed_id in sorted(list(failed_ids)):
                work_id, task, budget, pattern, pattern_idx_inner, behavior_log_prob = work_items_map[failed_id]
                failure_experience = {
                    "work_id": work_id,
                    "task_description": task['question'], 
                    "budget": budget,
                    "action": pattern_idx_inner, 
                    "is_correct": False,
                    "actual_cost": budget,
                    "planning_feasible": False,
                    "behavior_log_prob": behavior_log_prob,
                    "reason": "worker_timeout_or_unretrieved_result"
                }
                offline_dataset.append(failure_experience)
        stop_event.set()
        monitor_thread.join(timeout=5)
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join()
    output_path = params['data']['offline_dataset_path']
    base_path, ext = os.path.splitext(output_path)
    if args.begin_with > 0 or args.num_samples != -1:
        batch_suffix = f"_batch_{start_idx}_{end_idx-1}"
        single_suffix = f"_pattern_{pattern_idx}_budget_{budget_idx}{batch_suffix}"
    else:
        single_suffix = f"_pattern_{pattern_idx}_budget_{budget_idx}"
    output_path = f"{base_path}{single_suffix}{ext}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in offline_dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"{len(offline_dataset)} experiences saved to {output_path}")
    print(f"Budget: {target_budget} (idx={budget_idx})")
    if incorrect_res_data:
        base_dir = os.path.dirname(output_path)
        incorrect_output_path = f"{base_path}_incorrect{single_suffix}.json"
        with open(incorrect_output_path, 'w', encoding='utf-8') as f:
            for entry in incorrect_res_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Incorrect experiences ({len(incorrect_res_data)} items) saved to {incorrect_output_path}")
    return output_path
def main(args):
    """Generates an offline dataset in parallel."""
    mp.set_start_method("spawn", force=True)
    config_loader = ConfigLoader()
    params = config_loader.params
    train_dataset = load_gsm8k_dataset(path=params['data']['gsm8k_train_path'])
    random.seed(params['training']['seed'])
    random.shuffle(train_dataset)
    total_available = len(train_dataset)
    start_idx = args.begin_with
    if start_idx >= total_available:
        raise ValueError(f"begin_with ({start_idx}) exceeds dataset size ({total_available})")
    if args.num_samples == -1:
        end_idx = total_available
    else:
        end_idx = min(start_idx + args.num_samples, total_available)
    train_dataset = train_dataset[start_idx:end_idx]
    actual_samples = len(train_dataset)
    print(f"Processing samples {start_idx} to {end_idx-1} ({actual_samples} samples) from dataset of {total_available} total samples")
    all_patterns = config_loader.patterns['patterns']
    num_patterns = len(all_patterns)
    behavior_log_prob = np.log(1.0 / num_patterns)
    budgets_to_test = np.linspace(*params['training']['budget_range'], num=args.num_budget_steps).tolist()
    manager = mp.Manager()
    heartbeat_dict = manager.dict()
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    incorrect_queue = mp.Queue()
    work_items = []
    work_items_map = {}
    for i, task in enumerate(train_dataset):
        for j, budget in enumerate(budgets_to_test):
            for k, pattern in enumerate(all_patterns):
                global_task_idx = start_idx + i
                work_id = f"task_{global_task_idx}_budget_{j}_pattern_{k}"
                work_item = (work_id, task, budget, pattern, k, behavior_log_prob)
                work_items.append(work_item)
                work_items_map[work_id] = work_item
    for item in work_items:
        work_queue.put(item)
    for _ in range(args.num_workers):
        work_queue.put(None)
    workers = []
    for _ in range(args.num_workers):
        p = mp.Process(target=worker, args=(work_queue, result_queue, heartbeat_dict, incorrect_queue))
        p.start()
        workers.append(p)
    TIMEOUT_SECONDS = 300
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_workers,
        args=(workers, heartbeat_dict, stop_event, TIMEOUT_SECONDS),
        daemon=True
    )
    monitor_thread.start()
    offline_dataset = []
    incorrect_res_data = []
    total_work_items = len(work_items)
    try:
        with tqdm(total=total_work_items, desc="Generating Dataset") as pbar:
            while pbar.n < total_work_items:
                try:
                    experience = result_queue.get(timeout=5)
                    offline_dataset.append(experience)
                    pbar.update(1)
                    try:
                        while True:
                            incorrect_exp = incorrect_queue.get_nowait()
                            incorrect_res_data.append(incorrect_exp)
                    except queue.Empty:
                        pass
                except queue.Empty:
                    if not any(p.is_alive() for p in workers):
                        break
    finally:
        print("\n--- Initiating Cleanup ---")
        try:
            while True:
                incorrect_exp = incorrect_queue.get_nowait()
                incorrect_res_data.append(incorrect_exp)
        except queue.Empty:
            pass
        successful_ids = {exp.get('work_id') for exp in offline_dataset if exp.get('work_id')}
        all_ids = set(work_items_map.keys())
        failed_ids = all_ids - successful_ids
        if failed_ids:
            print(f"[Cleanup] Found {len(failed_ids)} incomplete tasks. Adding them as negative experiences.")
            for failed_id in sorted(list(failed_ids)):
                work_id, task, budget, pattern, pattern_idx, behavior_log_prob = work_items_map[failed_id]
                failure_experience = {
                    "work_id": work_id,
                    "task_description": task['question'], 
                    "budget": budget,
                    "action": pattern_idx, 
                    "is_correct": False,
                    "actual_cost": budget,
                    "planning_feasible": False,
                    "behavior_log_prob": behavior_log_prob,
                    "reason": "worker_timeout_or_unretrieved_result"
                }
                offline_dataset.append(failure_experience)
                incorrect_failure_experience = {
                    "work_id": work_id,
                    "task_description": task['question'],
                    "ground_truth_answer": task['answer'],
                    "budget": budget,
                    "action": pattern_idx,
                    "pattern_name": pattern['name'],
                    "is_correct": False,
                    "actual_cost": budget,
                    "planning_feasible": False,
                    "behavior_log_prob": behavior_log_prob,
                    "execution_history": [],
                    "full_context": {"history": ["Worker timeout or unretrieved result"]},
                    "reason": "worker_timeout_or_unretrieved_result"
                }
                incorrect_res_data.append(incorrect_failure_experience)
        failed_log_path = os.path.join(os.path.dirname(params['data']['offline_dataset_path']), "failed_tasks_log.jsonl")
        if args.begin_with > 0 or args.num_samples != -1:
            failed_log_dir = os.path.dirname(failed_log_path)
            batch_suffix = f"_batch_{start_idx}_{end_idx-1}"
            failed_log_path = os.path.join(failed_log_dir, f"failed_tasks_log{batch_suffix}.jsonl")
        if os.path.exists(failed_log_path):
            print(f"[Cleanup] Removing old failure log file: {failed_log_path}")
            os.remove(failed_log_path)
        stop_event.set()
        monitor_thread.join(timeout=5)
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join()
    output_path = params['data']['offline_dataset_path']
    if args.begin_with > 0 or args.num_samples != -1:
        base_path, ext = os.path.splitext(output_path)
        batch_suffix = f"_batch_{start_idx}_{end_idx-1}"
        output_path = f"{base_path}{batch_suffix}{ext}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in offline_dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"\nOffline dataset generation complete. {len(offline_dataset)} experiences saved to {output_path}")
    if incorrect_res_data:
        base_dir = os.path.dirname(output_path)
        if args.begin_with > 0 or args.num_samples != -1:
            batch_suffix = f"_batch_{start_idx}_{end_idx-1}"
            incorrect_output_path = os.path.join(base_dir, f"incorrect_res_data{batch_suffix}.json")
        else:
            incorrect_output_path = os.path.join(base_dir, "incorrect_res_data.json")
        os.makedirs(os.path.dirname(incorrect_output_path), exist_ok=True)
        with open(incorrect_output_path, 'w', encoding='utf-8') as f:
            for entry in incorrect_res_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Incorrect experiences ({len(incorrect_res_data)} items) saved to {incorrect_output_path}")
    else:
        print("No incorrect experiences to save.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline dataset for RL training in parallel.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of tasks. -1 for all.")
    parser.add_argument("--begin_with", type=int, default=0, help="Starting index for batch processing (0-based).")
    parser.add_argument("--num_budget_steps", type=int, default=5, help="Number of budget levels per task.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel worker processes.")
    parser.add_argument("--single_pattern", type=int, default=None, help="Generate only specific pattern (0-3)")
    parser.add_argument("--single_budget", type=int, default=None, help="Generate only specific budget (0-4)")
    args = parser.parse_args()
    if args.single_pattern is not None and args.single_budget is not None:
        generate_cache_single_pattern_budget(args, args.single_pattern, args.single_budget)
    else:
        main(args)