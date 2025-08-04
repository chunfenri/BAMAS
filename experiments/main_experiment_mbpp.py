"""
Main Experiment: Full Pipeline Inference with Trained High-Level Policy (Parallel Version)
Evaluates the trained system on MBPP test dataset across different budgets using multiprocessing.
"""
import torch
import random
import os
import json
import argparse
import sys
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import torch.multiprocessing as mp
import queue
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eapae_agent_sys.utils.config_loader import ConfigLoader
from eapae_agent_sys.data_processing.mbpp_loader import load_mbpp_dataset, prepare_mbpp_for_training
from eapae_agent_sys.planning.semantic_filter import SemanticFilter
from eapae_agent_sys.planning.low_level_instantiator import LowLevelInstantiator
from eapae_agent_sys.execution.execution_engine import ExecutionEngine
from eapae_agent_sys.planning.high_level_policy import HighLevelPolicy
from eapae_agent_sys.utils.llm_api import LLM_API
from eapae_agent_sys.utils.evaluation import evaluate_mbpp_success
def monitor_workers(workers: list, heartbeat_dict: dict, stop_event: threading.Event, timeout: int):
    """
    A thread that monitors worker processes and terminates them if they hang.
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
def worker(work_queue, result_queue, heartbeat_dict, model_state_dict, device_type):
    """Worker process to run a single experiment."""
    worker_pid = os.getpid()
    heartbeat_dict[worker_pid] = time.time()
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config_loader = ConfigLoader(
        agent_library_file="0_agent_library_mbpp.yml",
        training_params_file="3_training_params_mbpp.yml"
    )
    params = config_loader.params
    llm_api = LLM_API()
    semantic_filter = SemanticFilter(config_loader, llm_api)
    low_level_instantiator = LowLevelInstantiator(config_loader)
    execution_engine = ExecutionEngine(config_loader, low_level_instantiator)
    embedding_dim = 384
    num_patterns = len(config_loader.patterns['patterns'])
    max_budget = params['training']['budget_range'][1]
    policy_network = HighLevelPolicy(
        embedding_dim=embedding_dim, 
        num_patterns=num_patterns, 
        max_budget=max_budget, 
        device=device
    ).to(device)
    if model_state_dict is not None:
        policy_network.load_state_dict(model_state_dict)
    policy_network.eval()
    all_patterns = config_loader.patterns['patterns']
    while True:
        work_item = work_queue.get()
        if work_item is None:
            break
        heartbeat_dict[worker_pid] = time.time()
        work_id, task, budget, experiment_metadata = work_item
        try:
            with torch.no_grad():
                task_description = [task['question']]
                budget_tensor = torch.tensor([budget], device=device, dtype=torch.float32)
                selected_action, log_prob, logits = policy_network(
                    task_description, budget_tensor, is_deterministic=True
                )
                pattern_idx = selected_action.item()
                selected_pattern = all_patterns[pattern_idx]
        except Exception as e:
            print(f"[Worker-{worker_pid}] Policy inference failed for task {task.get('task_id', 'unknown')}: {e}")
            pattern_idx = 0
            selected_pattern = all_patterns[pattern_idx]
        is_correct = False
        actual_cost = 0.0
        planning_feasible = False
        inference_successful = False
        final_context = None
        try:
            candidate_agents = semantic_filter.filter_candidates(task['question'], collaboration_pattern=selected_pattern)
            if not candidate_agents:
                final_context = {"history": ["Planning failed: Semantic filter returned no candidate agents."]}
            else:
                agent_pool, is_feasible = low_level_instantiator.solve(
                    pattern_or_name=selected_pattern,
                    budget=budget,
                    candidate_agent_ids=candidate_agents
                )
                planning_feasible = is_feasible
                if is_feasible:
                    test_cases = task.get('test_cases', [])
                    enhanced_task_description = task['question']
                    if test_cases:
                        test_cases_str = '\n'.join(test_cases)
                        enhanced_task_description += f"\n\nTest cases for reference:\n{test_cases_str}"
                    final_context, actual_cost = execution_engine.execute_hybrid(
                        agent_pool, budget, enhanced_task_description, 
                        collaboration_pattern=selected_pattern,
                        dataset_type="mbpp"
                    )
                    success_code = evaluate_mbpp_success(final_context, task['answer'], task.get('test_cases', []))
                    is_correct = success_code == 1
                    inference_successful = True
                else:
                    final_context = {"history": ["Planning failed: No feasible agent team could be formed for the given budget and pattern."]}
        except Exception as e:
            print(f"[Worker-{worker_pid}] Inference failed: {e}")
            actual_cost = budget
            is_correct = False
            final_context = {"history": [f"Execution failed: {str(e)}"]}
        result = {
            "work_id": work_id,
            "task_id": task.get('task_id', 'unknown'),
            "task_description": task['question'],
            "ground_truth_answer": task.get('answer', ''),
            "budget": budget,
            "action": pattern_idx,
            "pattern_name": selected_pattern['name'],
            "is_correct": is_correct,
            "actual_cost": actual_cost,
            "planning_feasible": planning_feasible,
            "inference_successful": inference_successful,
            "timestamp": datetime.now().isoformat(),
            **experiment_metadata
        }
        result_queue.put(result)
def main():
    parser = argparse.ArgumentParser(description="Main experiment: Parallel full pipeline inference on MBPP test set")
    parser.add_argument("--model_path", type=str, help="Path to trained high-level policy model")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of test samples to use (-1 for all)")
    parser.add_argument("--num_budget_steps", type=int, default=5, help="Number of budget levels (default: 3 for MBPP)")
    parser.add_argument("--beginwith", type=int, default=0, help="Task index to start from (0-based)")
    parser.add_argument("--output_dir", type=str, default="experiments/results/mbpp", help="Output directory for results")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel worker processes")
    args = parser.parse_args()
    print("="*80)
    print("MAIN EXPERIMENT: PARALLEL FULL PIPELINE INFERENCE (MBPP DATASET)")
    print("="*80)
    mp.set_start_method("spawn", force=True)
    config_loader = ConfigLoader(
        agent_library_file="0_agent_library_mbpp.yml",
        training_params_file="3_training_params_mbpp.yml"
    )
    params = config_loader.params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = params.get('training', {}).get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw_test_dataset = load_mbpp_dataset("data/mbpp", split="test")
    test_dataset = []
    for sample in raw_test_dataset:
        processed_sample = prepare_mbpp_for_training(sample)
        test_dataset.append(processed_sample)
    if args.beginwith > 0:
        test_dataset = test_dataset[args.beginwith:]
    if args.num_samples != -1:
        test_dataset = test_dataset[:args.num_samples]
    model_state_dict = None
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(params['outputs']['checkpoints_dir'], "high_level_policy_mbpp_offline_best.pth")
    if os.path.exists(model_path):
        print(f"Loading trained MBPP model from: {model_path}")
        model_state_dict = torch.load(model_path, map_location='cpu')
    else:
        print(f"Warning: MBPP model not found at {model_path}")
    budgets = [500.0, 875.0, 1250.0, 1625.0, 2000.0]
    print(f"Testing with MBPP budgets: {budgets}")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manager = mp.Manager()
    heartbeat_dict = manager.dict()
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    work_items = []
    work_items_map = {}
    for task_idx, task in enumerate(test_dataset):
        for budget_idx, budget in enumerate(budgets):
            work_id = f"exp_{task_idx + args.beginwith}_budget_{budget_idx}"
            experiment_metadata = {
                "experiment_id": f"exp_{timestamp}",
                "task_idx": task_idx + args.beginwith,
                "budget_idx": budget_idx,
                "total_budgets": len(budgets),
                "total_tasks": len(test_dataset)
            }
            work_item = (work_id, task, budget, experiment_metadata)
            work_items.append(work_item)
            work_items_map[work_id] = work_item
    for item in work_items:
        work_queue.put(item)
    for _ in range(args.num_workers):
        work_queue.put(None)
    workers = []
    device_type = str(device).split(':')[0]
    for _ in range(args.num_workers):
        p = mp.Process(target=worker, args=(work_queue, result_queue, heartbeat_dict, model_state_dict, device_type))
        p.start()
        workers.append(p)
    TIMEOUT_SECONDS = 500
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_workers,
        args=(workers, heartbeat_dict, stop_event, TIMEOUT_SECONDS),
        daemon=True
    )
    monitor_thread.start()
    all_results = []
    total_work_items = len(work_items)
    try:
        with tqdm(total=total_work_items, desc="Running Experiments") as pbar:
            while pbar.n < total_work_items:
                try:
                    result = result_queue.get(timeout=5)
                    all_results.append(result)
                    pbar.set_postfix({
                        "Pattern": result['action'],
                        "Success": result['is_correct'],
                        "Budget": f"{result['budget']:.0f}"
                    })
                    pbar.update(1)
                except queue.Empty:
                    if not any(p.is_alive() for p in workers):
                        break
    finally:
        print("\n--- Initiating Cleanup ---")
        successful_ids = {result.get('work_id') for result in all_results if result.get('work_id')}
        all_ids = set(work_items_map.keys())
        failed_ids = all_ids - successful_ids
        if failed_ids:
            print(f"[Cleanup] Found {len(failed_ids)} incomplete tasks. Adding them as failure records.")
            for failed_id in sorted(list(failed_ids)):
                work_id, task, budget, experiment_metadata = work_items_map[failed_id]
                failure_result = {
                    "work_id": work_id,
                    "task_id": task.get('task_id', 'unknown'),
                    "task_description": task['question'],
                    "ground_truth_answer": task.get('answer', ''),
                    "budget": budget,
                    "action": 0,
                    "pattern_name": "unknown",
                    "is_correct": False,
                    "actual_cost": budget,
                    "planning_feasible": False,
                    "inference_successful": False,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "worker_timeout_or_failure",
                    **experiment_metadata
                }
                all_results.append(failure_result)
        stop_event.set()
        monitor_thread.join(timeout=5)
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join()
    results_file = os.path.join(args.output_dir, f"experiment_results_{timestamp}.jsonl")
    print(f"\nSaving detailed results to: {results_file}")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    summary_file = os.path.join(args.output_dir, f"experiment_summary_{timestamp}.json")
    print(f"Generating summary statistics: {summary_file}")
    stats_by_budget = {}
    stats_by_pattern = {}
    for budget in budgets:
        budget_results = [r for r in all_results if r['budget'] == budget]
        total = len(budget_results)
        correct = sum(1 for r in budget_results if r['is_correct'])
        feasible = sum(1 for r in budget_results if r['planning_feasible'])
        avg_cost = np.mean([r['actual_cost'] for r in budget_results])
        stats_by_budget[f"budget_{budget:.0f}"] = {
            "total_samples": total,
            "accuracy": correct / total if total > 0 else 0,
            "planning_feasible_rate": feasible / total if total > 0 else 0,
            "average_cost": avg_cost
        }
    all_patterns = config_loader.patterns['patterns']
    for pattern_idx in range(len(all_patterns)):
        pattern_results = [r for r in all_results if r['action'] == pattern_idx]
        total = len(pattern_results)
        correct = sum(1 for r in pattern_results if r['is_correct'])
        feasible = sum(1 for r in pattern_results if r['planning_feasible'])
        stats_by_pattern[f"pattern_{pattern_idx}"] = {
            "pattern_name": all_patterns[pattern_idx]['name'],
            "total_samples": total,
            "accuracy": correct / total if total > 0 else 0,
            "planning_feasible_rate": feasible / total if total > 0 else 0,
            "selection_frequency": total / len(all_results) if len(all_results) > 0 else 0
        }
    total_results = len(all_results)
    overall_accuracy = sum(1 for r in all_results if r['is_correct']) / total_results if total_results > 0 else 0
    overall_feasible_rate = sum(1 for r in all_results if r['planning_feasible']) / total_results if total_results > 0 else 0
    summary = {
        "experiment_info": {
            "timestamp": timestamp,
            "model_path": model_path,
            "total_tasks": len(test_dataset),
            "budgets_tested": budgets,
            "total_experiments": total_results,
            "num_workers": args.num_workers
        },
        "overall_performance": {
            "accuracy": overall_accuracy,
            "planning_feasible_rate": overall_feasible_rate
        },
        "performance_by_budget": stats_by_budget,
        "performance_by_pattern": stats_by_pattern
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {total_results}")
    print(f"Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"Planning feasible rate: {overall_feasible_rate:.3f} ({overall_feasible_rate*100:.1f}%)")
    print(f"\nPerformance by Budget:")
    for budget_key, stats in stats_by_budget.items():
        budget_val = budget_key.replace('budget_', '')
        print(f"  Budget {budget_val}: {stats['accuracy']:.3f} accuracy, {stats['planning_feasible_rate']:.3f} feasible")
    print(f"\nPattern Selection Frequency:")
    for pattern_key, stats in stats_by_pattern.items():
        pattern_name = pattern_key.replace('pattern_', '')
        print(f"  {pattern_name}: {stats['selection_frequency']:.3f} ({stats['selection_frequency']*100:.1f}%)")
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary: {summary_file}")
    print("="*80)
if __name__ == "__main__":
    main() 