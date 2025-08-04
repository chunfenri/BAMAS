import json
import re
import glob
import os
import ast
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Union, List, Literal, Any, Dict
import yaml
import numpy as np
def load_mbpp_dataset(path: str, split: Union[Literal['train'], Literal['test'], Literal['val'], Literal['all']] = 'train') -> List[Dict[str, Any]]:
    """
    Loads the MBPP dataset from .jsonl file and splits according to task_id ranges.
    Args:
        path: The path to the dataset .jsonl file
        split: The split to load ('train', 'test', 'val', 'all')
    Returns:
        A list of dictionaries, where each dictionary represents a sample.
    """
    if split in ['train', 'test', 'val']:
        processed_file = f"data/processed/mbpp/{split}.jsonl"
        if os.path.exists(processed_file):
            return _load_mbpp_jsonl(processed_file)
        else:
            print(f"Warning: Processed dataset {processed_file} not found.")
            return _create_mbpp_split_on_fly(path, split)
    if path.endswith('.jsonl'):
        return _load_mbpp_jsonl(path)
    else:
        print(f"Error: MBPP dataset path must be a .jsonl file, got: {path}")
        return []
def _load_mbpp_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load MBPP dataset from .jsonl file"""
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    dataset.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num} in {path}: {e}")
    except FileNotFoundError:
        print(f"Error: MBPP dataset file not found at {path}")
        return []
    return dataset
def _create_mbpp_split_on_fly(path: str, split: str) -> List[Dict[str, Any]]:
    """Create MBPP split on-the-fly based on task_id ranges"""
    all_data = _load_mbpp_jsonl(path)
    if split == 'train':
        task_id_range = (601, 974)
    elif split == 'test':
        task_id_range = (11, 510)
    elif split == 'val':
        task_id_range = (511, 600)
    else:
        return all_data
    filtered_data = []
    for sample in all_data:
        task_id = sample.get('task_id')
        if task_id and task_id_range[0] <= task_id <= task_id_range[1]:
            filtered_data.append(sample)
    return filtered_data
def get_mbpp_difficulty(task_id: int) -> str:
    """
    Assign difficulty level based on task_id.
    This is a simple heuristic - could be improved with actual complexity analysis.
    """
    if task_id <= 200:
        return "simple"
    elif task_id <= 600:
        return "medium"
    else:
        return "complex"
def count_mbpp_complexity(code: str) -> int:
    """
    Estimate code complexity by counting various code constructs.
    Returns an estimated complexity score.
    """
    if not code:
        return 1
    complexity = 1
    complexity += code.count('for ') * 2
    complexity += code.count('while ') * 2
    complexity += code.count('if ') 
    complexity += code.count('elif ')
    complexity += code.count('def ') * 3
    complexity += code.count('class ') * 5
    complexity += code.count('try:') * 2
    complexity += code.count('import ') 
    complexity += code.count('lambda ') * 2
    complexity += code.count('    ') // 4
    return max(1, complexity)
def extract_mbpp_answer(code: str) -> str:
    """
    Extract the main function or the final result from MBPP code.
    For MBPP, we typically want the entire function definition plus any necessary imports.
    """
    if not code:
        return ""
    code = code.replace('\r\n', '\n').replace('\r', '\n').strip()
    try:
        import ast
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    import re
    markdown_patterns = [
        r'```(?:python)?\s*\n(.*?)\n```',
        r'```python\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in markdown_patterns:
        markdown_match = re.search(pattern, code, re.DOTALL)
        if markdown_match:
            extracted_code = markdown_match.group(1).strip()
            try:
                ast.parse(extracted_code)
                return extracted_code
            except SyntaxError:
                continue
    llm_response_match = re.search(
        r'LLM RESPONSE START.*?```(?:python)?\s*\n(.*?)\n```.*?LLM RESPONSE END',
        code, re.DOTALL
    )
    if llm_response_match:
        extracted_code = llm_response_match.group(1).strip()
        try:
            ast.parse(extracted_code)
            return extracted_code
        except SyntaxError:
            pass
    lines = code.split('\n')
    import_lines = []
    function_lines = []
    in_function = False
    for line in lines:
        line_stripped = line.strip()
        if (line_stripped.startswith('import ') or 
            line_stripped.startswith('from ') and ' import ' in line_stripped):
            import_lines.append(line_stripped)
        elif line_stripped.startswith('def '):
            in_function = True
            function_lines.append(line)
        elif in_function:
            if line_stripped and not line.startswith(' ') and not line.startswith('\t'):
                if not (line_stripped.startswith('import ') or 
                       line_stripped.startswith('from ') and ' import ' in line_stripped):
                    break
                else:
                    import_lines.append(line_stripped)
            else:
                function_lines.append(line)
    if function_lines:
        if import_lines:
            result_lines = import_lines + [''] + function_lines
        else:
            result_lines = function_lines
        final_code = '\n'.join(result_lines)
        try:
            ast.parse(final_code)
            return final_code
        except SyntaxError:
            pass
    return code
def validate_python_syntax(code: str) -> bool:
    """
    Validate Python syntax of the given code.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
def execute_mbpp_code(code: str, test_cases: List[str], timeout_seconds: int = 10) -> Dict[str, Any]:
    """
    Execute MBPP code against test cases and return results.
    Args:
        code: The Python code to execute
        test_cases: List of test case strings (assert statements)
        timeout_seconds: Timeout for execution
    Returns:
        Dictionary with execution results
    """
    results = {
        'syntax_valid': False,
        'tests_passed': 0,
        'total_tests': len(test_cases),
        'success': False,
        'error': None,
        'execution_time': 0
    }
    if not validate_python_syntax(code):
        results['error'] = "Syntax Error"
        return results
    results['syntax_valid'] = True
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code + '\n\n')
            for test_case in test_cases:
                f.write(test_case + '\n')
            temp_file = f.name
        import time
        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            results['execution_time'] = time.time() - start_time
            if result.returncode == 0:
                results['tests_passed'] = len(test_cases)
                results['success'] = True
            else:
                results['error'] = result.stderr or result.stdout
        except subprocess.TimeoutExpired:
            results['error'] = "Execution Timeout"
    except Exception as e:
        results['error'] = str(e)
    finally:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
    return results
def prepare_mbpp_for_training(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares an MBPP dataset sample for training by standardizing the format.
    Args:
        sample: Raw MBPP dataset sample
    Returns:
        Processed sample compatible with training pipeline
    """
    task_id = sample.get('task_id', 0)
    processed = {
        'question': sample.get('text', ''),
        'answer': sample.get('code', ''),
        'task_id': task_id,
        'test_cases': sample.get('test_list', []),
        'test_setup_code': sample.get('test_setup_code', ''),
        'challenge_tests': sample.get('challenge_test_list', []),
        'difficulty': get_mbpp_difficulty(task_id),
        'extracted_answer': extract_mbpp_answer(sample.get('code', '')),
        'estimated_complexity': count_mbpp_complexity(sample.get('code', ''))
    }
    return processed
def get_difficulty_sampler(dataset: List[Dict], difficulty_field: str = 'difficulty'):
    """
    Groups MBPP data by difficulty for curriculum learning.
    Args:
        dataset: List of MBPP samples
        difficulty_field: Field name containing difficulty information
    Returns:
        Dictionary mapping difficulty levels to sample lists
    """
    difficulty_groups = defaultdict(list)
    for sample in dataset:
        difficulty = sample.get(difficulty_field, 'simple')
        difficulty_groups[difficulty].append(sample)
    for difficulty, samples in difficulty_groups.items():
        pass  # Group samples by difficulty
    return dict(difficulty_groups)
def MBPP_get_predict(answer_text: str) -> str:
    """
    Extract predicted code from answer text.
    For MBPP, this typically means extracting the function definition.
    """
    return extract_mbpp_answer(answer_text)
def MBPP_is_correct(predicted_code: str, expected_code: str, test_cases: List[str] = None) -> bool:
    """
    Check if predicted code is correct for MBPP.
    This involves running the code against test cases.
    """
    if not predicted_code or not test_cases:
        return False
    results = execute_mbpp_code(predicted_code, test_cases)
    return results['success'] and results['tests_passed'] == results['total_tests']
def compare_mbpp_code(code1: str, code2: str) -> bool:
    """
    Compare two pieces of code for functional equivalence.
    This is a simplified version - could be enhanced with more sophisticated analysis.
    """
    if not code1 or not code2:
        return False
    normalized_code1 = normalize_code(code1)
    normalized_code2 = normalize_code(code2)
    return normalized_code1 == normalized_code2
def normalize_code(code: str) -> str:
    """
    Normalize code for comparison by removing whitespace differences.
    """
    if not code:
        return ""
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines) 