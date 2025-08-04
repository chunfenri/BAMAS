import re
from typing import Dict
import yaml
def extract_answer(text: str) -> str:
    """
    Extracts the final numerical answer from a string.
    It prioritizes various answer formats in order of certainty.
    """
    final_answer_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if final_answer_match:
        return final_answer_match.group(1).strip()
    latex_boxed_patterns = [
        r'\\?\(\s*\\boxed\{(-?\d+(?:\.\d+)?)\}\s*\\?\)',
        r'\\boxed\{(-?\d+(?:\.\d+)?)\}',
        r'\\\[\s*\\boxed\{(-?\d+(?:\.\d+)?)\}\s*\\\]'
    ]
    for pattern in latex_boxed_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    math_patterns = [
        r'\$\$\s*(-?\d+(?:\.\d+)?)\s*\$\$',
        r'(?<!\$)\$\s*(-?\d+(?:\.\d+)?)\s*\$(?!\$)'
    ]
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    all_angle_matches = re.findall(r'<<([^>]+)>>\s*(-?\d+(?:\.\d+)?)?', text)
    if all_angle_matches:
        last_match = all_angle_matches[-1]
        if last_match[1]:
            return last_match[1].strip()
        else:
            inner_match = re.search(r'(-?\d+(?:\.\d+)?)$', last_match[0])
            if inner_match:
                return inner_match.group(1).strip()
    all_digit_matches = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if all_digit_matches:
        return all_digit_matches[-1].strip()
    return text.strip()
def extract_math_answer(text: str) -> str:
    """
    Specialized forMATHdataset answer extraction，handle more complexLaTeXformat。
    MATHdataset answer format is more complex，including fractions、radicals、expressions etc。
    """
    boxed_patterns = [
        r'\\boxed\{([^{}]+)\}',
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\boxed\{(.+?)\}',
        r'\\?\(\s*\\boxed\{([^}]+)\}\s*\\?\)',
        r'\\?\[\s*\\boxed\{([^}]+)\}\s*\\?\]',
    ]
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            answer = matches[-1].strip()
            answer = answer.replace('\\text', '').replace('{', '').replace('}', '')
            return answer
    final_answer_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if final_answer_match:
        return final_answer_match.group(1).strip()
    answer_patterns = [
        r'[Tt]he answer is\s*(.+?)(?:\.|$)',
        r'[Aa]nswer:\s*(.+?)(?:\.|$)',
        r'[Ff]inal answer:\s*(.+?)(?:\.|$)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return extract_answer(text)
def normalize_math_answer(answer: str) -> str:
    """
    NormalizeMATHdataset answer format，for comparison。
    handle common mathematical expression equivalent forms。
    """
    if not answer:
        return ""
    answer = answer.strip()
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = answer.replace('\\', '')
    frac_pattern = r'frac\{([^}]+)\}\{([^}]+)\}'
    answer = re.sub(frac_pattern, r'(\1)/(\2)', answer)
    if re.match(r'^\([^()]+\)$', answer):
        answer = answer[1:-1]
    replacements = {
        'pi': 'π',
        'infty': '∞',
        'sqrt': '√',
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    return answer
def compare_math_answers(answer1: str, answer2: str) -> bool:
    """
    Compare twoMATHwhether answers are equivalent。
    consider mathematical expression equivalence。
    """
    norm1 = normalize_math_answer(answer1)
    norm2 = normalize_math_answer(answer2)
    if norm1 == norm2:
        return True
    try:
        def eval_fraction(s):
            if '/' in s:
                parts = s.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            return float(s)
        val1 = eval_fraction(norm1)
        val2 = eval_fraction(norm2)
        return abs(val1 - val2) < 1e-9
    except:
        pass
    equiv_patterns = [
        (r'^(\d+)/(\d+)$', r'^\1/\2$'),
        (r'^(\d+)$', r'^\1\.0*$'),
    ]
    for pattern1, pattern2 in equiv_patterns:
        if re.match(pattern1, norm1) and re.match(pattern2, norm2):
            return True
        if re.match(pattern2, norm1) and re.match(pattern1, norm2):
            return True
    return False
def evaluate_success(final_context: dict, expected_answer_str: str) -> int:
    """
    Evaluates if the final answer from the structured history matches the expected answer.
    """
    if not final_context or not isinstance(final_context, dict) or 'history' not in final_context:
        print("Evaluation result: Error  (final_contextincorrect format or empty)")
        return -1
    history = final_context['history']
    if not history or not isinstance(history, list):
        print("Evaluation result: Error  (historyempty or incorrect format)")
        return -1
    last_step = history[-1]
    if not isinstance(last_step, dict) or 'result' not in last_step:
        print("Evaluation result: Error  (last step history record format incorrect)")
        return -1
    system_answer_raw = last_step['result']
    if "ERROR" in str(system_answer_raw):
         print(f"Evaluation result: Error  (error encountered during execution: {system_answer_raw})")
         return -1
    system_answer_cleaned = extract_answer(str(system_answer_raw))
    expected_answer_cleaned = extract_answer(expected_answer_str)
    print("\n" + "="*20 + " Answer evaluation details " + "="*20)
    print(f"Original system answer: {system_answer_raw}")
    print(f"Cleaned system answer: {system_answer_cleaned}")
    print(f"Original expected answer: {expected_answer_str}")
    print(f"Cleaned expected answer: {expected_answer_cleaned}")
    try:
        system_float = float(system_answer_cleaned)
        expected_float = float(expected_answer_cleaned)
        if system_float == expected_float:
            print("Evaluation result: Correct  (numerical match)")
            return 1
    except (ValueError, TypeError):
        pass
    if system_answer_cleaned == expected_answer_cleaned:
        print("Evaluation result: Correct  (string exact match)")
        return 1
    else:
        print("Evaluation result: Error ")
        return 0
def evaluate_math_success(final_context: dict, expected_answer_str: str) -> int:
    """
    Specialized forMATHdataset evaluation success rate，UsestandardofMATHanswer comparisonlogic。
    """
    if not final_context or not isinstance(final_context, dict) or 'history' not in final_context:
        print("Evaluation result: Error  (final_contextincorrect format or empty)")
        return -1
    history = final_context['history']
    if not history or not isinstance(history, list):
        print("Evaluation result: Error  (historyempty or incorrect format)")
        return -1
    last_step = history[-1]
    if not isinstance(last_step, dict) or 'result' not in last_step:
        print("Evaluation result: Error  (last step history record format incorrect)")
        return -1
    system_answer_raw = last_step['result']
    if "ERROR" in str(system_answer_raw):
         print(f"Evaluation result: Error  (error encountered during execution: {system_answer_raw})")
         return -1
    try:
        from eapae_agent_sys.data_processing.math_loader import MATH_get_predict, MATH_is_correct
        if isinstance(system_answer_raw, dict) and 'result' in system_answer_raw:
            system_answer_text = system_answer_raw['result']
        else:
            system_answer_text = str(system_answer_raw)
        if '\\boxed' not in system_answer_text:
            system_answer_text = f'\\boxed{{{system_answer_text}}}'
        system_prediction = MATH_get_predict(system_answer_text)
        print("\n" + "="*20 + " MATHAnswer evaluation details " + "="*20)
        print(f"Original system answer: {system_answer_raw}")
        print(f"Extracted answer text: {system_answer_text}")
        print(f"MATHExtracted prediction: {system_prediction}")
        print(f"Standard answer: {expected_answer_str}")
        is_correct = MATH_is_correct(system_prediction, expected_answer_str)
        if is_correct:
            print("Evaluation result: Correct  (MATHstandard match)")
            return 1
        else:
            print("Evaluation result: Error ")
            return 0
    except ImportError:
        print("Warning: Unable to importMATHstandard functionuse original method")
        system_answer_cleaned = extract_math_answer(str(system_answer_raw))
        expected_answer_cleaned = extract_math_answer(expected_answer_str)
        print("\n" + "="*20 + " MATHAnswer evaluation details " + "="*20)
        print(f"Original system answer: {system_answer_raw}")
        print(f"Cleaned system answer: {system_answer_cleaned}")
        print(f"Original expected answer: {expected_answer_str}")
        print(f"Cleaned expected answer: {expected_answer_cleaned}")
        if compare_math_answers(system_answer_cleaned, expected_answer_cleaned):
            print("Evaluation result: Correct  (MATHformat match)")
            return 1
        else:
            print("Evaluation result: Error ")
            return 0
def get_evaluator(dataset_type: str = "gsm8k"):
    """
    Return corresponding evaluation function based on dataset type。
    """
    if dataset_type.lower() == "math":
        return evaluate_math_success
    elif dataset_type.lower() == "mbpp":
        return evaluate_mbpp_success
    else:
        return evaluate_success
def evaluate_mbpp_success(final_context: dict, expected_answer_str: str, test_cases: list = None) -> int:
    """
    Specialized forMBPPdataset evaluation success rate，judge by executing code and running test cases。
    """
    if not final_context or not isinstance(final_context, dict) or 'history' not in final_context:
        print("Evaluation result: Error  (final_contextincorrect format or empty)")
        return -1
    history = final_context['history']
    if not history or not isinstance(history, list):
        print("Evaluation result: Error  (historyempty or incorrect format)")
        return -1
    last_step = history[-1]
    if not isinstance(last_step, dict) or 'result' not in last_step:
        print("Evaluation result: Error  (last step history record format incorrect)")
        return -1
    system_answer_raw = last_step['result']
    if "ERROR" in str(system_answer_raw):
         print(f"Evaluation result: Error  (error encountered during execution: {system_answer_raw})")
         return -1
    try:
        from eapae_agent_sys.data_processing.mbpp_loader import MBPP_get_predict, MBPP_is_correct, execute_mbpp_code
        if isinstance(system_answer_raw, dict) and 'result' in system_answer_raw:
            system_code = system_answer_raw['result']
        else:
            system_code = str(system_answer_raw)
        system_prediction = MBPP_get_predict(system_code)
        print("\n" + "="*20 + " MBPPCode evaluation details " + "="*20)
        print(f"Original system answer: {system_answer_raw}")
        print(f"Number of test cases: {len(test_cases) if test_cases else 0}")
        if test_cases and len(test_cases) > 0:
            execution_results = execute_mbpp_code(system_prediction, test_cases)
            print(f"Syntax valid: {execution_results['syntax_valid']}")
            print(f"Tests passed: {execution_results['tests_passed']}/{execution_results['total_tests']}")
            print(f"execution time: {execution_results['execution_time']:.2f}s")
            if execution_results['error']:
                print(f"Execution error: {execution_results['error']}")
            if execution_results['success']:
                print("Evaluation result: Correct  (All test cases passed)")
                return 1
            else:
                print("Evaluation result: Error  (Test cases not all passed)")
                return 0
        else:
            is_correct = MBPP_is_correct(system_prediction, expected_answer_str, test_cases)
            if is_correct:
                print("Evaluation result: Correct  (MBPPstandard match)")
                return 1
            else:
                print("Evaluation result: Error ")
                return 0
    except ImportError as e:
        print(f"Warning: Unable to importMBPPevaluation function: {e}")
        print("Use simple string comparison asfallback")
        if isinstance(system_answer_raw, dict) and 'result' in system_answer_raw:
            system_code = system_answer_raw['result']
        else:
            system_code = str(system_answer_raw)
        print("\n" + "="*20 + " MBPPCode evaluation details (Fallback) " + "="*20)
        print(f"Original system answer: {system_answer_raw}")
        system_code_normalized = normalize_code_for_comparison(system_code)
        expected_code_normalized = normalize_code_for_comparison(expected_answer_str)
        if system_code_normalized == expected_code_normalized:
            print("Evaluation result: Correct  (string match)")
            return 1
        else:
            print("Evaluation result: Error ")
            return 0
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        return -1
def normalize_code_for_comparison(code: str) -> str:
    """
    Normalize code for comparison by removing whitespace differences.
    """
    if not code:
        return ""
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines) 