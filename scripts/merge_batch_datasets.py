"""
Merge multiple batch dataset files into a single offline dataset.
Usage: python merge_batch_datasets.py --input_pattern "data/offline_rl_dataset_batch_*.jsonl" --output "data/offline_rl_dataset.jsonl"
"""
import argparse
import glob
import json
import os
from collections import defaultdict
def main():
    parser = argparse.ArgumentParser(description="Merge batch dataset files into single dataset")
    parser.add_argument("--input_pattern", type=str, required=True, 
                       help="Glob pattern for batch files (e.g., 'data/processed/offline_rl_dataset_batch_*.jsonl')")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path for merged dataset")
    parser.add_argument("--verify", action="store_true",
                       help="Verify no duplicate work_ids across batches")
    args = parser.parse_args()
    batch_files = sorted(glob.glob(args.input_pattern))
    if not batch_files:
        return
    for f in batch_files:
        print(f"  - {f}")
    merged_data = []
    work_ids_seen = set()
    duplicate_count = 0
    for batch_file in batch_files:
        batch_count = 0
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                work_id = entry.get('work_id')
                if args.verify and work_id in work_ids_seen:
                    duplicate_count += 1
                    continue
                if work_id:
                    work_ids_seen.add(work_id)
                merged_data.append(entry)
                batch_count += 1
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Total entries: {len(merged_data)}")
    if args.verify:
        print(f"Verification: {len(merged_data)} total entries processed")
    print(f"Output saved to: {args.output}")
    success_count = sum(1 for entry in merged_data if entry.get('is_correct', False))
    feasible_count = sum(1 for entry in merged_data if entry.get('planning_feasible', False))
    print(f"\nDataset Statistics:")
    print(f"  Success rate: {success_count}/{len(merged_data)} ({100*success_count/len(merged_data):.1f}%)")
    print(f"  Planning feasible rate: {feasible_count}/{len(merged_data)} ({100*feasible_count/len(merged_data):.1f}%)")
if __name__ == "__main__":
    main() 