import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from eapae_agent_sys.utils.config_loader import ConfigLoader
from eapae_agent_sys.planning.high_level_policy import HighLevelPolicy
class OfflineRLDataset(Dataset):
    """PyTorch Dataset for loading the pre-generated offline RL experiences."""
    def __init__(self, file_path, use_ranking=True, use_weighted_ranking=False, config_loader=None):
        self.experiences = []
        self.use_ranking = use_ranking
        self.use_weighted_ranking = use_weighted_ranking
        if config_loader:
            self.reward_params = config_loader.params['rewards']
        else:
            default_config = ConfigLoader()
            self.reward_params = default_config.params['rewards']
        all_experiences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_experiences.append(json.loads(line))
        if use_weighted_ranking:
            self.experiences = self._apply_weighted_ranking(all_experiences)
        elif use_ranking:
            self.experiences = self._apply_ranking_filter(all_experiences)
        else:
            self.experiences = all_experiences
    def _apply_ranking_filter(self, all_experiences):
        """
        For eachpieces(task, budget)combination，selectoptimalpatterndata
        prioritize accuracy，consider efficiency only when accuracy is similar
        """
        groups = {}
        for exp in all_experiences:
            key = (exp['task_description'], exp['budget'])
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)
        filtered_experiences = []
        ranking_stats = {
            'total_groups': len(groups),
            'patterns_per_group': [],
            'best_pattern_distribution': {}
        }
        for (task, budget), group in groups.items():
            for exp in group:
                exp['computed_reward'] = self._compute_reward(exp)
            correct_exps = [exp for exp in group if exp['is_correct']]
            incorrect_exps = [exp for exp in group if not exp['is_correct']]
            if correct_exps:
                best_exp = max(correct_exps, key=lambda x: x['computed_reward'])
            else:
                best_exp = max(group, key=lambda x: x['computed_reward'])
            filtered_experiences.append(best_exp)
            ranking_stats['patterns_per_group'].append(len(group))
            best_pattern = best_exp['action']
            ranking_stats['best_pattern_distribution'][best_pattern] = \
                ranking_stats['best_pattern_distribution'].get(best_pattern, 0) + 1
        print(f"  Total (task, budget) groups: {ranking_stats['total_groups']}")
        print(f"  Best pattern distribution (accuracy-first):")
        for pattern, count in ranking_stats['best_pattern_distribution'].items():
            pct = 100 * count / ranking_stats['total_groups']
            print(f"    Pattern {pattern}: {count} ({pct:.1f}%)")
        return filtered_experiences
    def _apply_weighted_ranking(self, all_experiences):
        """
        UseAlldata，but calculate eachsamplesofactualrewardfor weighted training
        """
        weighted_experiences = []
        for exp in all_experiences:
            exp['computed_reward'] = self._compute_reward(exp)
            weighted_experiences.append(exp)
        rewards = [exp['computed_reward'] for exp in weighted_experiences]
        print(f"  Total samples: {len(weighted_experiences)}")
        print(f"  Mean reward: {np.mean(rewards):.2f}")
        print(f"  Positive rewards: {sum(1 for r in rewards if r > 0)} ({100*sum(1 for r in rewards if r > 0)/len(rewards):.1f}%)")
        pattern_rewards = {}
        for exp in weighted_experiences:
            pattern = exp['action']
            if pattern not in pattern_rewards:
                pattern_rewards[pattern] = []
            pattern_rewards[pattern].append(exp['computed_reward'])
        print(f"  Reward by pattern:")
        for pattern, pattern_reward_list in pattern_rewards.items():
            mean_reward = np.mean(pattern_reward_list)
            max_reward = np.max(pattern_reward_list)
            min_reward = np.min(pattern_reward_list)
            print(f"    Pattern {pattern}: mean {mean_reward:.2f}, max {max_reward:.2f}, min {min_reward:.2f}, count {len(pattern_reward_list)}")
        return weighted_experiences
    def _compute_reward(self, exp):
        """
        calculate singlepiecesexperienceofreward，Usein paperofcomposite reward function：
        R_final(τ) = w_task · R_task + w_budget · R_budget
        """
        reward_params = self.reward_params
        if not exp['planning_feasible']:
            return reward_params.get('shaping_infeasible', reward_params['failure_penalty'])
        if exp['is_correct']:
            R_task = reward_params['success_reward']
        else:
            R_task = reward_params['failure_penalty']
        if exp['actual_cost'] > exp['budget']:
            R_budget = -reward_params['overflow_penalty']
        else:
            if exp['is_correct']:
                R_budget = reward_params['efficiency_multiplier'] * (1 - (exp['actual_cost'] / exp['budget']))
            else:
                R_budget = 0.0
        w_task = reward_params.get('task_weight', 1.0)
        w_budget = reward_params.get('budget_weight', 1.0)
        return w_task * R_task + w_budget * R_budget
    def __len__(self):
        return len(self.experiences)
    def __getitem__(self, idx):
        exp = self.experiences[idx]
        if self.use_weighted_ranking:
            return (
                exp['task_description'],
                exp['budget'],
                exp['action'],
                exp['is_correct'],
                exp['actual_cost'],
                exp['planning_feasible'],
                exp['behavior_log_prob'],
                exp['computed_reward']
            )
        else:
            return (
                exp['task_description'],
                exp['budget'],
                exp['action'],
                exp['is_correct'],
                exp['actual_cost'],
                exp['planning_feasible'],
                exp['behavior_log_prob']
            )
def calculate_rewards(is_correct, actual_cost, budget, planning_feasible, reward_params):
    """
    Calculates rewards using the paper's composite reward function:
    R_final(τ) = w_task · R_task + w_budget · R_budget
    """
    if not planning_feasible:
        return reward_params.get('shaping_infeasible', reward_params['failure_penalty'])
    if is_correct:
        R_task = reward_params['success_reward']
    else:
        R_task = reward_params['failure_penalty']
    if actual_cost > budget:
        R_budget = -reward_params['overflow_penalty']
    else:
        if is_correct:
            R_budget = reward_params['efficiency_multiplier'] * (1 - (actual_cost / budget))
        else:
            R_budget = 0.0
    w_task = reward_params.get('task_weight', 1.0)
    w_budget = reward_params.get('budget_weight', 1.0)
    return w_task * R_task + w_budget * R_budget
def train_epoch(policy, optimizer, dataloader, device, params, epoch=1, total_epochs=1, scheduler=None, eval_dataset=None):
    """
    Helper function to train a single epoch using REINFORCE algorithm as described in the paper.
    """
    epoch_network_actions = []
    epoch_gt_actions = []
    epoch_gt_rewards = []
    epoch_gt_success_rates = []
    epoch_gt_budgets = []
    epoch_gt_costs = []
    epoch_losses = []
    entropy_beta = params['training'].get('entropy_beta', 0.001)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} (β={entropy_beta})")
    for batch in pbar:
        if len(batch) == 8:
            tasks, budgets, actions, is_corrects, costs, feasibles, behavior_log_probs, computed_rewards = batch
            computed_rewards_tensor = computed_rewards.to(device, dtype=torch.float32)
        else:
            tasks, budgets, actions, is_corrects, costs, feasibles, behavior_log_probs = batch
            computed_rewards = torch.tensor([
                calculate_rewards(ic, ac, b, pf, params['rewards'])
                for ic, ac, b, pf in zip(is_corrects, costs, budgets, feasibles)
            ], dtype=torch.float32)
            computed_rewards_tensor = computed_rewards.to(device)
        budgets_tensor = budgets.to(device, dtype=torch.float32)
        actions_tensor = actions.to(device, dtype=torch.long)
        with torch.no_grad():
            network_actions, _, _ = policy(list(tasks), budgets_tensor, is_deterministic=False)
        _, log_probs, logits = policy(list(tasks), budgets_tensor, forced_action_idx=actions_tensor)
        probs = torch.softmax(logits, dim=-1)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()
        policy_loss = -(log_probs * computed_rewards_tensor).mean()
        total_loss = policy_loss - entropy_beta * entropy
        loss_info = {
            "Loss": f"{total_loss.item():.4f}",
            "Policy": f"{policy_loss.item():.4f}",
            "Entropy": f"{entropy.item():.4f}",
            "Mean_Reward": f"{computed_rewards_tensor.mean().item():.2f}"
        }
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_network_actions.extend(network_actions.cpu().tolist())
        epoch_gt_actions.extend(actions_tensor.cpu().tolist())
        epoch_gt_success_rates.extend(is_corrects.tolist())
        epoch_gt_budgets.extend(budgets.cpu().tolist())
        epoch_gt_costs.extend(costs.tolist())
        epoch_losses.append(total_loss.item())
        epoch_gt_rewards.extend(computed_rewards_tensor.cpu().tolist())
        pbar.set_postfix(loss_info)
    if eval_dataset is not None:
        print(f"\n{'='*60}")
        print(f"UNIFIED EVALUATION (Oracle Dataset)")
        print(f"{'='*60}")
        policy.eval()
        with torch.no_grad():
            eval_network_actions = []
            eval_gt_actions = []
            eval_gt_rewards = []
            eval_gt_success_rates = []
            eval_gt_costs = []
            eval_gt_budgets = []
            eval_dataloader = DataLoader(eval_dataset, batch_size=1000, shuffle=False, num_workers=4)
            for batch in eval_dataloader:
                if len(batch) == 7:
                    tasks, budgets, actions, is_corrects, costs, feasibles, behavior_log_probs = batch
                else:
                    tasks, budgets, actions, is_corrects, costs, feasibles, behavior_log_probs, computed_rewards = batch
                budgets_tensor = budgets.to(device, dtype=torch.float32)
                actions_tensor = actions.to(device, dtype=torch.long)
                network_actions, _, _ = policy(list(tasks), budgets_tensor, is_deterministic=False)
                rewards = torch.tensor([
                    calculate_rewards(ic, ac, b, pf, params['rewards'])
                    for ic, ac, b, pf in zip(is_corrects, costs, budgets, feasibles)
                ], device=device, dtype=torch.float32)
                eval_network_actions.extend(network_actions.cpu().tolist())
                eval_gt_actions.extend(actions_tensor.cpu().tolist())
                eval_gt_rewards.extend(rewards.cpu().tolist())
                eval_gt_success_rates.extend(is_corrects.tolist())
                eval_gt_costs.extend(costs.tolist())
                eval_gt_budgets.extend(budgets.cpu().tolist())
        policy.train()
        total_samples = len(eval_network_actions)
        eval_network_actions_tensor = torch.tensor(eval_network_actions)
        eval_gt_actions_tensor = torch.tensor(eval_gt_actions)
        eval_gt_unique_actions, eval_gt_action_counts = torch.unique(eval_gt_actions_tensor, return_counts=True)
        for action, count in zip(eval_gt_unique_actions, eval_gt_action_counts):
            print(f"  Pattern {action.item()}: {count.item()} ({100*count.item()/total_samples:.1f}%)")
        eval_unique_actions, eval_action_counts = torch.unique(eval_network_actions_tensor, return_counts=True)
        for action, count in zip(eval_unique_actions, eval_action_counts):
            print(f"  Pattern {action.item()}: {count.item()} ({100*count.item()/total_samples:.1f}%)")
        agreement_mask = (eval_network_actions_tensor == eval_gt_actions_tensor)
        agreement_rate = agreement_mask.float().mean().item()
        print(f"\nUnified Performance Analysis:")
        print(f"  Agreement with Oracle (Accuracy): {agreement_rate:.3f} ({100*agreement_rate:.1f}%)")
        agreed_success_rate = 0.0
        agreed_out_of_budget_rate = 0.0
        agreed_count = 0
        agreed_mean_reward = 0.0
        if agreement_mask.sum() > 0:
            agreed_success = [eval_gt_success_rates[i] for i in range(len(eval_gt_success_rates)) if agreement_mask[i]]
            agreed_costs = [eval_gt_costs[i] for i in range(len(eval_gt_costs)) if agreement_mask[i]]
            agreed_budgets = [eval_gt_budgets[i] for i in range(len(eval_gt_budgets)) if agreement_mask[i]]
            agreed_rewards = [eval_gt_rewards[i] for i in range(len(eval_gt_rewards)) if agreement_mask[i]]
            agreed_mean_reward = np.mean(agreed_rewards)
            agreed_out_of_budget_count = sum(1 for cost, budget in zip(agreed_costs, agreed_budgets) if cost > budget)
            agreed_out_of_budget_rate = agreed_out_of_budget_count / len(agreed_rewards) if len(agreed_rewards) > 0 else 0
            agreed_success_rate = np.mean(agreed_success)
            agreed_count = len(agreed_rewards)
            print(f"  When network agrees with Oracle (n={agreed_count}):")
            print(f"    Mean reward: {agreed_mean_reward:.4f}")
            print(f"    Success rate: {agreed_success_rate:.3f} ({100*agreed_success_rate:.1f}%)")
            print(f"    Out of budget: {agreed_out_of_budget_count}/{agreed_count} ({100*agreed_out_of_budget_rate:.1f}%)")
        else:
            print(f"  When network agrees with Oracle: No agreements found")
        if agreed_mean_reward > 0:
            normalized_reward = min(agreed_mean_reward / 100.0, 1.0)
            current_performance_score = agreement_rate * 0.7 + normalized_reward * 0.3
        else:
            current_performance_score = agreement_rate
        print(f"  Performance score: {current_performance_score:.4f} (Accuracy: {agreement_rate:.3f}, Reward: {agreed_mean_reward:.2f})")
        epoch_mean_loss = np.mean(epoch_losses)
        current_lr = scheduler.get_last_lr()[0] if scheduler else 0.0
        print(f"  Mean training loss: {epoch_mean_loss:.4f}")
        print(f"  Current learning rate: {current_lr:.6f}")
        return {
            'epoch': epoch,
            'performance_score': current_performance_score,
            'agreement_rate': agreement_rate,
            'agreed_mean_reward': agreed_mean_reward,
            'agreed_success_rate': agreed_success_rate,
            'agreed_out_of_budget_rate': agreed_out_of_budget_rate,
            'agreed_count': agreed_count,
            'mean_loss': epoch_mean_loss,
            'total_samples': total_samples,
            'current_lr': current_lr,
            'dataset_success_rate': np.mean(eval_gt_success_rates),
            'dataset_out_of_budget_rate': sum(1 for cost, budget in zip(eval_gt_costs, eval_gt_budgets) if cost > budget) / len(eval_gt_costs)
        }
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch} SUMMARY")
    print(f"{'='*60}")
    total_samples = len(epoch_network_actions)
    epoch_gt_actions_tensor = torch.tensor(epoch_gt_actions)
    gt_unique_actions, gt_action_counts = torch.unique(epoch_gt_actions_tensor, return_counts=True)
    for action, count in zip(gt_unique_actions, gt_action_counts):
        print(f"  Pattern {action.item()}: {count.item()} ({100*count.item()/total_samples:.1f}%)")
    epoch_network_actions_tensor = torch.tensor(epoch_network_actions)
    unique_actions, action_counts = torch.unique(epoch_network_actions_tensor, return_counts=True)
    for action, count in zip(unique_actions, action_counts):
        print(f"  Pattern {action.item()}: {count.item()} ({100*count.item()/total_samples:.1f}%)")
    gt_out_of_budget_count = sum(1 for cost, budget in zip(epoch_gt_costs, epoch_gt_budgets) if cost > budget)
    gt_out_of_budget_rate = gt_out_of_budget_count / total_samples
    print(f"\nBudget analysis:")
    print(f"  GT data out of budget: {gt_out_of_budget_count}/{total_samples} ({100*gt_out_of_budget_rate:.1f}%)")
    agreement_mask = (epoch_network_actions_tensor == epoch_gt_actions_tensor)
    agreement_rate = agreement_mask.float().mean().item()
    print(f"\nNetwork behavior analysis:")
    print(f"  Agreement with ground truth: {agreement_rate:.3f} ({100*agreement_rate:.1f}%)")
    current_performance_score = agreement_rate
    agreed_success_rate = 0.0
    agreed_out_of_budget_rate = 0.0
    agreed_count = 0
    agreed_mean_reward = 0.0
    if agreement_mask.sum() > 0:
        agreed_success = [epoch_gt_success_rates[i] for i in range(len(epoch_gt_success_rates)) if agreement_mask[i]]
        agreed_costs = [epoch_gt_costs[i] for i in range(len(epoch_gt_costs)) if agreement_mask[i]]
        agreed_budgets = [epoch_gt_budgets[i] for i in range(len(epoch_gt_budgets)) if agreement_mask[i]]
        agreed_rewards = [epoch_gt_rewards[i] for i in range(len(epoch_gt_rewards)) if agreement_mask[i]]
        agreed_mean_reward = np.mean(agreed_rewards)
        agreed_out_of_budget_count = sum(1 for cost, budget in zip(agreed_costs, agreed_budgets) if cost > budget)
        agreed_out_of_budget_rate = agreed_out_of_budget_count / len(agreed_rewards) if len(agreed_rewards) > 0 else 0
        agreed_success_rate = np.mean(agreed_success)
        agreed_count = len(agreed_rewards)
        print(f"  When network agrees with GT (n={agreed_count}):")
        print(f"    Mean reward: {agreed_mean_reward:.4f}")
        print(f"    Success rate: {agreed_success_rate:.3f} ({100*agreed_success_rate:.1f}%)")
        print(f"    Out of budget: {agreed_out_of_budget_count}/{agreed_count} ({100*agreed_out_of_budget_rate:.1f}%)")
    else:
        print(f"  When network agrees with GT: No agreements found")
    if agreed_mean_reward > 0:
        normalized_reward = min(agreed_mean_reward / 100.0, 1.0)
        current_performance_score = agreement_rate * 0.7 + normalized_reward * 0.3
    print(f"  Performance score: {current_performance_score:.4f} (Agreement: {agreement_rate:.3f}, Reward: {agreed_mean_reward:.2f})")
    epoch_mean_loss = np.mean(epoch_losses)
    current_lr = scheduler.get_last_lr()[0] if scheduler else 0.0
    print(f"  Mean training loss: {epoch_mean_loss:.4f}")
    print(f"  Current learning rate: {current_lr:.6f}")
    return {
        'epoch': epoch,
        'performance_score': current_performance_score,
        'agreement_rate': agreement_rate,
        'agreed_mean_reward': agreed_mean_reward,
        'agreed_success_rate': agreed_success_rate,
        'agreed_out_of_budget_rate': agreed_out_of_budget_rate,
        'agreed_count': agreed_count,
        'mean_loss': epoch_mean_loss,
        'total_samples': total_samples,
        'current_lr': current_lr,
        'dataset_success_rate': np.mean(epoch_gt_success_rates),
        'dataset_out_of_budget_rate': gt_out_of_budget_rate
    }
def get_dataset_config(dataset_name):
    """Get configuration files for different datasets."""
    dataset_configs = {
        'gsm8k': {
            'agent_library_file': None,
            'training_params_file': None,
            'display_name': 'GSM8K'
        },
        'math': {
            'agent_library_file': '0_agent_library_math.yml',
            'training_params_file': '3_training_params_math.yml',
            'display_name': 'MATH'
        },
        'mbpp': {
            'agent_library_file': '0_agent_library_mbpp.yml',
            'training_params_file': '3_training_params_mbpp.yml',
            'display_name': 'MBPP'
        }
    }
    if dataset_name.lower() not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(dataset_configs.keys())}")
    return dataset_configs[dataset_name.lower()]
def main(args):
    """Main function for offline training."""
    dataset_config = get_dataset_config(args.dataset_name)
    print(f"Training Mode: REINFORCE Algorithm (Policy Gradient + Entropy Regularization)")
    total_epochs = args.epochs
    if dataset_config['agent_library_file'] and dataset_config['training_params_file']:
        config_loader = ConfigLoader(
            agent_library_file=dataset_config['agent_library_file'],
            training_params_file=dataset_config['training_params_file']
        )
    else:
        config_loader = ConfigLoader()
    params = config_loader.params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = params['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_path = params['data']['offline_dataset_path']
    if not os.path.exists(dataset_path):
        print(f"Error: Offline dataset not found at {dataset_path}")
        print("Please run 'scripts/3_generate_training_cache.py' first.")
        return
    dataset = OfflineRLDataset(dataset_path, use_ranking=False, use_weighted_ranking=False, config_loader=config_loader)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    embedding_dim = 384
    num_patterns = len(config_loader.patterns['patterns'])
    max_budget = params['training']['budget_range'][1]
    policy = HighLevelPolicy(embedding_dim, num_patterns, max_budget, device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=params['training']['learning_rate'])
    lr_decay_epochs = params['training'].get('lr_decay_epochs', 20)
    lr_decay_factor = params['training'].get('lr_decay_factor', 0.6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_factor)
    print(f"Learning Rate Scheduler: StepLR(step_size={lr_decay_epochs}, gamma={lr_decay_factor})")
    print("\n" + "="*60)
    print("EVALUATING INITIAL NETWORK PERFORMANCE")
    print("="*60)
    eval_dataset = dataset
    sample_size = len(eval_dataset)
    sample_indices = random.sample(range(len(eval_dataset)), sample_size)
    sample_data = [eval_dataset[i] for i in sample_indices]
    with torch.no_grad():
        policy.eval()
        sample_tasks = [data[0] for data in sample_data]
        sample_budgets = torch.tensor([data[1] for data in sample_data], device=device, dtype=torch.float32)
        sample_actions = torch.tensor([data[2] for data in sample_data], device=device, dtype=torch.long)
        sample_is_corrects = [data[3] for data in sample_data]
        sample_costs = [data[4] for data in sample_data]
        sample_feasibles = [data[5] for data in sample_data]
        sample_behavior_log_probs = torch.tensor([data[6] for data in sample_data], device=device, dtype=torch.float32)
        initial_actions, initial_log_probs, initial_logits = policy(sample_tasks, sample_budgets, is_deterministic=False)
        logits_mean = initial_logits.mean(dim=0)
        logits_std = initial_logits.std(dim=0)
        print(f"Mean logits per pattern: {logits_mean.cpu().numpy()}")
        print(f"Std logits per pattern:  {logits_std.cpu().numpy()}")
        print(f"Max - Min logits per sample:")
        logits_range = initial_logits.max(dim=1)[0] - initial_logits.min(dim=1)[0]
        print(f"  Range mean: {logits_range.mean().item():.4f}, std: {logits_range.std().item():.4f}")
        for i in range(min(5, initial_logits.size(0))):
            range_val = logits_range[i].item()
            print(f"  Sample {i}: range = {range_val:.4f}")
        print(f"=====================================\n")
        gt_rewards = [calculate_rewards(ic, ac, b, pf, params['rewards']) 
                     for ic, ac, b, pf in zip(sample_is_corrects, sample_costs, sample_budgets.cpu(), sample_feasibles)]
        print(f"Sample size: {sample_size}")
        unique_actions, action_counts = torch.unique(sample_actions, return_counts=True)
        for action, count in zip(unique_actions, action_counts):
            print(f"  Pattern {action.item()}: {count.item()} ({100*count.item()/sample_size:.1f}%)")
        initial_unique_actions, initial_action_counts = torch.unique(initial_actions, return_counts=True)
        for action, count in zip(initial_unique_actions, initial_action_counts):
            print(f"  Pattern {action.item()}: {count.item()} ({100*count.item()/sample_size:.1f}%)")
        print(f"\nGround truth reward statistics (Oracle Performance):")
        gt_rewards_tensor = torch.tensor(gt_rewards)
        print(f"  Mean reward: {gt_rewards_tensor.mean().item():.4f}")
        print(f"  Std reward: {gt_rewards_tensor.std().item():.4f}")
        print(f"  Min reward: {gt_rewards_tensor.min().item():.4f}")
        print(f"  Max reward: {gt_rewards_tensor.max().item():.4f}")
        success_rate = sum(sample_is_corrects) / sample_size
        feasible_rate = sum(sample_feasibles) / sample_size
        print(f"  Success rate: {success_rate:.3f} ({100*success_rate:.1f}%)")
        print(f"  Planning feasible rate: {feasible_rate:.3f} ({100*feasible_rate:.1f}%)")
        positive_rewards = [r for r in gt_rewards if r > 0]
        negative_rewards = [r for r in gt_rewards if r <= 0]
        initial_agreement = (initial_actions == sample_actions).float().mean().item()
        print(f"\nInitial network vs Oracle agreement: {initial_agreement:.3f} ({100*initial_agreement:.1f}%)")
        policy.train()
    print("="*60)
    print("="*60 + "\n")
    best_mean_reward = float('-inf')
    best_epoch = -1
    best_model_state = None
    best_stats = {}
    checkpoints_dir = params['outputs']['checkpoints_dir']
    os.makedirs(checkpoints_dir, exist_ok=True)
    dataset_suffix = f"_{args.dataset_name.lower()}" if args.dataset_name.lower() != "gsm8k" else ""
    model_path = os.path.join(checkpoints_dir, f"high_level_policy_offline_best{dataset_suffix}.pth")
    stats_path = os.path.join(checkpoints_dir, f"best_model_stats{dataset_suffix}.json")
    for epoch in range(args.epochs):
        current_epoch = epoch + 1
        print(f"\nEpoch {current_epoch}/{args.epochs}")
        epoch_stats = train_epoch(
            policy, optimizer, dataloader, device, params, 
            epoch=current_epoch, total_epochs=args.epochs,
            scheduler=scheduler
        )
        if epoch_stats['agreed_mean_reward'] > best_mean_reward:
            best_mean_reward = epoch_stats['agreed_mean_reward']
            best_epoch = current_epoch
            best_model_state = policy.state_dict().copy()
            best_stats = epoch_stats.copy()
            print(f"   NEW BEST MODEL! Epoch {best_epoch}, Mean Reward: {best_mean_reward:.4f}")
            torch.save(best_model_state, model_path)
            with open(stats_path, 'w') as f:
                json.dump(best_stats, f, indent=2)
            print(f"  Model saved to {model_path}")
        else:
            print(f"  Current best: Epoch {best_epoch}, Mean Reward: {best_mean_reward:.4f}")
        scheduler.step()
    if best_model_state is not None:
        print(f" Best model location: {model_path}")
        print(f" Best model stats: {stats_path}")
        print(f" Final best performance:")
    else:
        print(f"  No valid model found during training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a policy using REINFORCE algorithm as described in the paper.")
    parser.add_argument("--dataset_name", type=str, default="mbpp", choices=["gsm8k", "math", "mbpp"], 
                       help="Dataset to train on (default: mbpp)")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    args = parser.parse_args()
    main(args) 
