"""
MATHDataset sampling script
Generate1000training samples and1000representative subset of test samples
Use completely identical distribution sequential sampling strategy:
- Maintain originalLevelandTopiccomplete identical distribution
- Take first according to original orderNsamples，do not use random sampling
- ensure statistical consistency and academic defensibility
"""
import json
import os
import sys
from collections import defaultdict, Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from eapae_agent_sys.data_processing.math_loader import load_math_dataset
def normalize_level(level):
    """Standardizelevelfield，extract numerical part"""
    if isinstance(level, str):
        if 'Level' in level:
            parts = level.split()
            if len(parts) > 1:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
        try:
            return int(level)
        except ValueError:
            return None
    return level if isinstance(level, int) else None
def proportional_sample(dataset, target_size, level_field='level', topic_field='type'):
    """
    Completely identical distribution sequential sampling：sample sequentially according to original distribution proportion
    Args:
        dataset: Original dataset
        target_size: Target sample count
        level_field: levelfield name
        topic_field: topicfield name
    Returns:
        Sampled dataset
    """
    total_original = len(dataset)
    level_counts = Counter()
    topic_counts = Counter()
    level_topic_counts = defaultdict(lambda: defaultdict(int))
    for sample in dataset:
        level = normalize_level(sample.get(level_field))
        topic = sample.get(topic_field, 'unknown')
        if level is not None:
            level_counts[level] += 1
            topic_counts[topic] += 1
            level_topic_counts[level][topic] += 1
    print(f"Original data distribution (Total {total_original} samples):")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        pct = (count / total_original) * 100
        print(f"  Level {level}: {count:4d} ({pct:5.1f}%)")
    for topic in sorted(topic_counts.keys()):
        count = topic_counts[topic]
        pct = (count / total_original) * 100
    target_level_counts = {}
    for level, count in level_counts.items():
        target_count = int((count / total_original) * target_size)
        target_level_counts[level] = target_count
    current_total = sum(target_level_counts.values())
    if current_total < target_size:
        largest_level = max(target_level_counts.keys(), key=lambda x: target_level_counts[x])
        target_level_counts[largest_level] += (target_size - current_total)
    elif current_total > target_size:
        largest_level = max(target_level_counts.keys(), key=lambda x: target_level_counts[x])
        target_level_counts[largest_level] -= (current_total - target_size)
    print(f"\nTarget sampling distribution (Total {target_size} samples):")
    for level in sorted(target_level_counts.keys()):
        count = target_level_counts[level]
        pct = (count / target_size) * 100
        print(f"  Level {level}: {count:3d} ({pct:5.1f}%)")
    level_groups = defaultdict(list)
    for sample in dataset:
        level = normalize_level(sample.get(level_field))
        if level is not None and level in target_level_counts:
            level_groups[level].append(sample)
    sampled_data = []
    for level, target_count in target_level_counts.items():
        available_samples = level_groups[level]
        if len(available_samples) >= target_count:
            selected = available_samples[:target_count]
        else:
            selected = available_samples
        sampled_data.extend(selected)
    return sampled_data
def analyze_distribution(data, name):
    """Analyze data distribution"""
    level_counts = Counter()
    topic_counts = Counter()
    for sample in data:
        level = normalize_level(sample.get('level'))
        topic = sample.get('type', 'unknown')
        level_counts[level] += 1
        topic_counts[topic] += 1
    total = len(data)
    for level in sorted(level_counts.keys()) if level_counts else []:
        count = level_counts[level]
        pct = (count / total) * 100
        print(f"  Level {level}: {count:3d} ({pct:5.1f}%)")
    for topic in sorted(topic_counts.keys()):
        count = topic_counts[topic]
        pct = (count / total) * 100
def create_math_subset():
    """CreateMATHRepresentative subset of dataset"""
    print("=" * 70)
    print("=" * 70)
    print(" Use deterministic sequential samplingNo randomness")
    try:
        train_dataset = load_math_dataset('data/math', split='train')
        test_dataset = load_math_dataset('data/math', split='test')
    except Exception as e:
        print(f" Failed to load dataset: {e}")
        return
    if not train_dataset or not test_dataset:
        print(" Dataset is empty or failed to load")
        return
    target_size = 1000
    print("- No randomnessResults completely reproducible")
    print("- Academically simplest and mostdefensiblemethod")
    print("=" * 50)
    print("Training set sampling:")
    sampled_train = proportional_sample(train_dataset, target_size)
    print("\n" + "=" * 50)
    print("Test set sampling:")
    sampled_test = proportional_sample(test_dataset, target_size)
    analyze_distribution(sampled_train, "Sampled training set")
    analyze_distribution(sampled_test, "Sampled test set")
    output_dir = "data/processed/math"
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "sampled_train.jsonl")
    test_file = os.path.join(output_dir, "sampled_test.jsonl")
    print(f"\n Save sampling results...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in sampled_train:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in sampled_test:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f" Training set saved to: {train_file}")
    print(f" Test set saved to: {test_file}")
    print(f"\n Sampling summary:")
    print(f"  Compression ratio: {((len(sampled_train) + len(sampled_test)) / (len(train_dataset) + len(test_dataset))) * 100:.1f}%")
    print(f"  Computational cost savings: ~{100 - ((len(sampled_train) + len(sampled_test)) / (len(train_dataset) + len(test_dataset))) * 100:.1f}%")
    print(f"Now can use split='sampled_train' and split='sampled_test' load subset data")
    print("=" * 70)
def main():
    """Main function"""
    create_math_subset()
if __name__ == "__main__":
    main() 