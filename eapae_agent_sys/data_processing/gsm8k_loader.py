import json
from collections import defaultdict
def load_gsm8k_dataset(path: str) -> list[dict]:
    """
    Loads a dataset from a .jsonl file.
    Args:
        path: The path to the .jsonl file.
    Returns:
        A list of dictionaries, where each dictionary represents a sample.
    """
    if not path.endswith('.jsonl'):
        raise ValueError("The dataset path must point to a .jsonl file.")
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {path}")
        return []
    return dataset
def get_difficulty_sampler(dataset: list[dict], levels: list[str]) -> dict[str, list[dict]]:
    """
    Groups a dataset by difficulty levels for curriculum learning.
    Args:
        dataset: The full dataset, loaded as a list of dicts.
        levels: A list of difficulty strings to sample from (e.g., ['easy', 'medium']).
    Returns:
        A dictionary where keys are difficulty levels and values are lists of
        samples corresponding to that level.
    """
    sampler = defaultdict(list)
    level_set = set(levels)
    for item in dataset:
        difficulty = item.get('difficulty')
        if difficulty in level_set:
            sampler[difficulty].append(item)
    for level in levels:
        if not sampler[level]:
            print(f"Warning: No samples found for difficulty level '{level}'.")
    return sampler 