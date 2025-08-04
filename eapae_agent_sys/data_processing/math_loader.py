import json
import re
import glob
import os
from collections import defaultdict
from typing import Union, List, Literal, Any, Dict
import yaml
import numpy as np
def load_math_dataset(path: str, split: Union[Literal['train'], Literal['test'], Literal['sampled_train'], Literal['sampled_test']] = 'train') -> List[Dict[str, str]]:
    """
    Loads the MATH dataset from directory structure or .jsonl file.
    Args:
        path: The path to the dataset directory or .jsonl file.
        split: The split to load ('train', 'test', 'sampled_train', 'sampled_test')
    Returns:
        A list of dictionaries, where each dictionary represents a sample.
    """
    if path.endswith('.jsonl'):
        return _load_math_jsonl(path)
    if split in ['sampled_train', 'sampled_test']:
        sampled_file = f"data/processed/math/{split}.jsonl"
        if os.path.exists(sampled_file):
            return _load_math_jsonl(sampled_file)
        else:
            print(f"Warning: Sampled dataset {sampled_file} not found.")
            original_split = 'train' if split == 'sampled_train' else 'test'
            return load_math_dataset(path, split=original_split)
    split_path = os.path.join(path, split)
    if not os.path.exists(split_path):
        if os.path.exists(path):
            json_files = glob.glob(os.path.join(path, "*.json"))
            if json_files:
                print(f"Loading from flat directory structure: {path}")
                return _load_math_flat_directory(path)
        print(f"Error: Dataset split path not found at {split_path}")
        return []
    category_paths = glob.glob(os.path.join(split_path, "*"))
    category_paths = sorted(category_paths)
    print("Number of categories: ", len(category_paths))
    total_data = []
    for category_path in category_paths:
        if os.path.isdir(category_path):  
            json_files = glob.glob(os.path.join(category_path, "*.json"))
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_data.append(data)
    print("Total number of questions: ", len(total_data))
    rng = np.random.default_rng(888)
    shuffled_data = list(rng.permutation(total_data))
    return shuffled_data
def _load_math_jsonl(path: str) -> List[Dict]:
    """Load MATH dataset from .jsonl file (legacy format)"""
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {path}")
        return []
    return dataset
def _load_math_flat_directory(path: str) -> List[Dict]:
    """Load MATH dataset from flat directory with JSON files"""
    dataset = []
    json_files = glob.glob(os.path.join(path, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                dataset.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return dataset
def _fix_fracs(string):
    """Fix fraction formatting in MATH answers"""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string
def _fix_a_slash_b(string):
    """Fix a/b fraction format"""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string
def _remove_right_units(string):
    """Remove units from the right side of answers"""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string
def _fix_sqrt(string):
    """Fix sqrt formatting"""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string
def _strip_string(string):
    """Strip and normalize MATH answer strings"""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string
def is_equiv(str1, str2, verbose=False):
    """Check if two MATH answers are equivalent"""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2
def last_boxed_only_string(string):
    """Extract the last \\boxed{} content from a string"""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval
def remove_boxed(s):
    """Remove \\boxed{} wrapper from answer"""
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
def MATH_get_predict(pred_str):
    """Extract prediction from model output using MATH standard logic"""
    if '\\boxed' in pred_str:
        pred = remove_boxed(last_boxed_only_string(pred_str))
        return pred.strip() if pred is not None else "0"
    elif('answer is ' in pred_str):
        pred = pred_str.split('answer is ')[-1].strip().rstrip(".")
        return pred.strip()
    elif len(pred_str) > 0:
        return pred_str[-1]
    else:
        return "A"
def MATH_is_correct(pred, reference):
    """Check if prediction is correct using MATH standard logic"""
    true_answer_str = remove_boxed(last_boxed_only_string(reference))
    if pred is not None and is_equiv(true_answer_str, pred):
        return True
    return False
def get_math_difficulty(level: int) -> str:
    """
    Maps MATH dataset level to difficulty category.
    Args:
        level: Integer level from MATH dataset (1-5)
    Returns:
        Difficulty string for curriculum learning
    """
    try:
        config_path = "configs/4_math_dataset_config.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        difficulty_mapping = config['difficulty_mapping']
        return difficulty_mapping.get(level, f"level_{level}")
    except:
        mapping = {1: "level_1", 2: "level_2", 3: "level_3", 4: "level_4", 5: "level_5"}
        return mapping.get(level, f"level_{level}")
def extract_math_answer(solution: str) -> str:
    """
    Extracts the final answer from MATH dataset solution using standard method.
    Args:
        solution: The solution string from MATH dataset
    Returns:
        Extracted answer string
    """
    return MATH_get_predict(solution)
def get_math_difficulty_sampler(dataset: list[dict], levels: list[int]) -> dict[str, list[dict]]:
    """
    Groups MATH dataset by difficulty levels for curriculum learning.
    Args:
        dataset: The full MATH dataset, loaded as a list of dicts.
        levels: A list of difficulty integers to sample from (e.g., [1, 2, 3]).
    Returns:
        A dictionary where keys are difficulty levels and values are lists of
        samples corresponding to that level.
    """
    sampler = defaultdict(list)
    level_set = set(levels)
    for item in dataset:
        level = item.get('level')
        if level in level_set:
            difficulty = get_math_difficulty(level)
            item['difficulty'] = difficulty
            sampler[difficulty].append(item)
    for level in levels:
        difficulty = get_math_difficulty(level)
        if not sampler[difficulty]:
            print(f"Warning: No samples found for difficulty level '{difficulty}' (level {level}).")
    return sampler
def count_math_steps(solution: str) -> int:
    """
    Estimates the number of reasoning steps in a MATH solution.
    Used for alternative difficulty assessment.
    Args:
        solution: The solution string
    Returns:
        Estimated number of steps
    """
    step_indicators = [
        r'\$[^$]+\$',
        r'\$\$[^$]+\$\$',
        r'\\begin\{align\}',
        r'\\begin\{equation\}',
        r'\n\n',
        r'Therefore',
        r'Hence',
        r'Thus',
        r'So',
    ]
    total_steps = 0
    for pattern in step_indicators:
        total_steps += len(re.findall(pattern, solution, re.IGNORECASE | re.DOTALL))
    return max(1, total_steps)
def prepare_math_for_training(sample: dict) -> dict:
    """
    Prepares a MATH dataset sample for training by standardizing the format.
    Args:
        sample: Raw MATH dataset sample
    Returns:
        Processed sample compatible with training pipeline
    """
    processed = {
        'question': sample.get('problem', ''),
        'answer': sample.get('solution', ''),
        'level': sample.get('level', 1),
        'type': sample.get('type', 'unknown'),
        'difficulty': get_math_difficulty(sample.get('level', 1)),
        'extracted_answer': extract_math_answer(sample.get('solution', '')),
        'estimated_steps': count_math_steps(sample.get('solution', ''))
    }
    return processed 