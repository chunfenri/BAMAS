"""
MBPPDataset split script
According to official specificationMBPPsplit dataset into training、test and validation sets
Split rules：
- Task IDs 1-10: few-shot prompting (not used for training)
- Task IDs 11-510: test set (500samples)  
- Task IDs 511-600: validation set (90samples)
- Task IDs 601-974: training set (374samples)
"""
import json
import os
import sys
from collections import defaultdict, Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from eapae_agent_sys.data_processing.mbpp_loader import load_mbpp_dataset, prepare_mbpp_for_training
def analyze_mbpp_distribution(data, name):
    """AnalyzeMBPPdata distribution"""
    task_ids = [sample.get('task_id', 0) for sample in data]
    difficulty_counts = Counter()
    for sample in data:
        processed = prepare_mbpp_for_training(sample)
        difficulty = processed.get('difficulty', 'unknown')
        difficulty_counts[difficulty] += 1
    total = len(data)
    print(f"Total samples: {total}")
    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        pct = (count / total) * 100
        print(f"  {difficulty}: {count:3d} ({pct:5.1f}%)")
def create_mbpp_splits():
    """CreateMBPPofficial split of dataset"""
    print("=" * 70)
    print("MBPPDataset split tool")
    print("=" * 70)
    raw_data_path = "data/mbpp/mbpp.jsonl"
    try:
        full_dataset = load_mbpp_dataset(raw_data_path, split='all')
    except Exception as e:
        print(f" Failed to load dataset: {e}")
        return
    if not full_dataset:
        print(" Dataset is empty or failed to load")
        return
    task_id_to_sample = {}
    for sample in full_dataset:
        task_id = sample.get('task_id')
        if task_id:
            task_id_to_sample[task_id] = sample
    print(f" ValidTask IDRange: {min(task_id_to_sample.keys())} - {max(task_id_to_sample.keys())}")
    splits = {
        'train': (601, 974),
        'test': (11, 510),
        'val': (511, 600),
        'few_shot': (1, 10)
    }
    for split_name, (start_id, end_id) in splits.items():
        expected_count = end_id - start_id + 1
    split_data = {}
    for split_name, (start_id, end_id) in splits.items():
        split_samples = []
        for task_id in range(start_id, end_id + 1):
            if task_id in task_id_to_sample:
                split_samples.append(task_id_to_sample[task_id])
        split_data[split_name] = split_samples
    for split_name, samples in split_data.items():
        if samples:
            analyze_mbpp_distribution(samples, f"{split_name.upper()}set")
    output_dir = "data/processed/mbpp"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n save split results to {output_dir}...")
    for split_name in ['train', 'test', 'val']:
        if split_name in split_data:
            split_file = os.path.join(output_dir, f"{split_name}.jsonl")
            with open(split_file, 'w', encoding='utf-8') as f:
                for sample in split_data[split_name]:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f" {split_name:9s}set saved to: {split_file}")
    if 'few_shot' in split_data:
        few_shot_file = os.path.join(output_dir, "few_shot.jsonl")
        with open(few_shot_file, 'w', encoding='utf-8') as f:
            for sample in split_data['few_shot']:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f" few_shotset saved to: {few_shot_file}")
    print(f"\n Split summary:")
    total_used = len(split_data['train']) + len(split_data['test']) + len(split_data['val'])
    print(f"  Total used: {total_used:3d} samples")
    print("Now can use split='train'/'test'/'val' load corresponding split data")
    print("=" * 70)
def main():
    """Main function"""
    create_mbpp_splits()
if __name__ == "__main__":
    main() 