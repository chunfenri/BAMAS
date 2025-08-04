from .base_agent import BaseAgent
from typing import Dict, Tuple
import traceback
import io
import sys
import re
def _extract_final_answer(text: str) -> str:
    """
    Extracts the final numerical answer from a string that might contain reasoning.
    It looks for a number following a specific '####' marker first, then falls back to the last number.
    """
    pattern = r'####\s*(-?[\d,]+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].replace(',', '')
    all_numbers = re.findall(r'(-?[\d,]+(?:\.\d+)?)', text)
    if all_numbers:
        valid_numbers = []
        for num in all_numbers:
            clean_num = num.replace(',', '')
            if clean_num.replace('.', '').replace('-', '').isdigit():
                valid_numbers.append(clean_num)
        if valid_numbers:
            return valid_numbers[-1]
    return "ERROR_PARSING"
class MathExecutorAgent(BaseAgent):
    """
    Execute a naturally described、single-step mathematical calculation。
    """
    def __init__(self, agent_config_id: str, config: Dict):
        super().__init__(agent_config_id, config)
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        sub_task = state.get("instruction") or state.get("sub_task") or state.get("expression") or state.get("problem") or "No sub-task provided."
        dataset_type = state.get("dataset_type", "gsm8k")
        is_detailed_mode = (
            "solve this problem:" in sub_task.lower() or
            "review and improve this solution:" in sub_task.lower() or
            "verify this solution and confirm" in sub_task.lower() or
            "step by step" in sub_task.lower() or
            len(sub_task) > 200
        )
        prompt = self._create_prompt(sub_task, dataset_type)
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        if is_detailed_mode:
            if dataset_type == "math":
                clean_response = self._extract_math_answer(response)
            else:
                clean_response = _extract_final_answer(response)
            output_dict = {
                "result": response,
                "numerical_answer": clean_response
            }
        else:
            if dataset_type == "math":
                clean_response = self._extract_math_answer(response)
            else:
                clean_response = _extract_final_answer(response)
            output_dict = {"result": clean_response}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, sub_task: str, dataset_type: str) -> str:
        """
        Creates a flexible prompt that adapts to different instruction types.
        Supports both simple calculations and detailed collaborative solving.
        """
        is_detailed_mode = (
            "solve this problem:" in sub_task.lower() or
            "review and improve this solution:" in sub_task.lower() or
            "verify this solution and confirm" in sub_task.lower() or
            len(sub_task) > 200
        )
        if is_detailed_mode:
            if "verify this solution" in sub_task.lower():
                if dataset_type == "math":
                    return (
                        "You are a mathematics expert verifying a colleague's solution. Check the solution carefully and provide confirmation.\n"
                        "Think through the problem step by step, show your reasoning, then give your final answer.\n"
                        "If you need to do calculations, show your work clearly - include all necessary steps but very brief.(The thinking step should not use LaTeX formatting, each step should be very short if the problem is difficult)\n"
                        "IMPORTANT: Read the question carefully and answer exactly what is being asked.\n"
                        "For competition math problems, you must express your final answer using LaTeX formatting.\n"
                        "End with your final answer enclosed in \\boxed{} (e.g., \\boxed{42} or \\boxed{\\frac{1}{2}}).\n"
                        "CRITICAL: Your \\boxed{} answer should contain ONLY the requested value (number/expression), without units or extra text.\n\n"
                        "--- EXAMPLE ---\n"
                        'Task: "Verify this solution and confirm the final answer: To solve x^2 = 16, we take the square root: x = 4."\n'
                        "Response:\n"
                        "Let me verify this solution step by step:\n\n"
                        "The equation is x^2 = 16.\n"
                        "Taking the square root of both sides: x = ±√16 = ±4\n\n"
                        "However, the given solution only provides x = 4, missing the negative solution.\n"
                        "The complete solution should be x = 4 or x = -4.\n\n"
                        "\\boxed{\\pm 4}\n\n"
                        "--- YOUR TASK ---\n"
                        f'Task: "{sub_task}"\n'
                        "Response:\n"
                    )
                else:
                    return (
                        "You are a mathematics expert verifying a colleague's solution. Check the solution carefully and provide confirmation.\n"
                        "Think through the problem step by step, show your reasoning, then give your final answer.\n"
                        "If you need to do calculations, show your work clearly - include all necessary steps but keep explanations brief.\n"
                        "IMPORTANT: Keep your final answer in the same units as mentioned in the problem. Don't convert units unless explicitly asked.\n"
                        "End with your final answer marked as: #### [numerical answer]\n\n"
                        "--- EXAMPLE ---\n"
                        'Task: "Verify this solution and confirm the final answer: The total cost is apples (4×$3=$12) plus oranges (6×$2=$12), so $12+$12=$24."\n'
                        "Response:\n"
                        "Let me verify this solution step by step:\n\n"
                        "First, let me check the apple calculation:\n"
                        "4 apples × $3 each = $12 ✓ This is correct.\n\n"
                        "Next, the orange calculation:\n"
                        "6 oranges × $2 each = $12 ✓ This is also correct.\n\n"
                        "Finally, the total:\n"
                        "$12 + $12 = $24 ✓ The arithmetic is right.\n\n"
                        "The solution is completely correct.\n"
                        "#### 24\n\n"
                        "--- YOUR TASK ---\n"
                        f'Task: "{sub_task}"\n'
                        "Response:\n"
                    )
            else:
                if dataset_type == "math":
                    return (
                        "You are a mathematics expert working in a collaborative team. Think through the problem step by step.\n"
                        "Show your reasoning process - include all necessary steps but very brief.(The thinking step should not use LaTeX formatting, each step should be very short if the problem is difficult)\n"
                        "Do any necessary calculations and explain your approach.\n"
                        "IMPORTANT: Read the question carefully and answer exactly what is being asked.\n"
                        "For competition math problems, you must express your final answer using LaTeX formatting.\n"
                        "End with your final answer enclosed in \\boxed{} (e.g., \\boxed{42} or \\boxed{\\frac{1}{2}}).\n"
                        "CRITICAL: Your \\boxed{} answer should contain ONLY the requested value (number/expression), without units or extra text.\n\n"
                        "--- EXAMPLE ---\n"
                        'Task: "Solve this problem: Find the value of x if 2x + 5 = 13."\n'
                        "Response:\n"
                        "I need to solve 2x + 5 = 13 for x.\n\n"
                        "Subtracting 5 from both sides:\n"
                        "2x + 5 - 5 = 13 - 5\n"
                        "2x = 8\n\n"
                        "Dividing both sides by 2:\n"
                        "x = 4\n\n"
                        "\\boxed{4}\n\n"
                        "--- YOUR TASK ---\n"
                        f'Task: "{sub_task}"\n'
                        "Response:\n"
                    )
                else:
                    return (
                        "You are a mathematics expert working in a collaborative team. Think through the problem step by step.\n"
                        "Show your reasoning process - include all necessary steps but keep explanations brief.\n"
                        "Do any necessary calculations and explain your approach.\n"
                        "IMPORTANT: Keep your final answer in the same units as mentioned in the problem. Don't convert units unless explicitly asked.\n"
                        "End with your final answer marked as: #### [numerical answer]\n\n"
                        "--- EXAMPLE ---\n"
                        'Task: "Solve this problem: Maria buys 4 apples at $3 each. How much does she spend?"\n'
                        "Response:\n"
                        "I need to calculate the total cost of 4 apples at $3 each.\n\n"
                        "Total cost = Number of apples × Price per apple\n"
                        "Total cost = 4 × $3 = $12\n\n"
                        "Maria spends $12 on apples.\n"
                        "#### 12\n\n"
                        "--- YOUR TASK ---\n"
                        f'Task: "{sub_task}"\n'
                        "Response:\n"
                    )
        else:
            if dataset_type == "math":
                return (
                    "You are a precise mathematical calculator. Solve the given problem step by step.\n"
                    "Show your work clearly - include all necessary steps but very brief.(The thinking step should not use LaTeX formatting, each step should be very short if the problem is difficult)\n"
                    "IMPORTANT: Read the question carefully and answer exactly what is being asked.\n"
                    "For competition math problems, you must express your final answer using LaTeX formatting.\n"
                    "IMPORTANT: Express your final answer using LaTeX formatting in \\boxed{} (e.g., \\boxed{42}).\n"
                    "CRITICAL: Your \\boxed{} answer should contain ONLY the requested value (number/expression), without units or extra text.\n\n"
                    f'Problem: "{sub_task}"\n'
                    "Response:\n"
                )
            else:
                return (
                    "You are a precise mathematical calculator. Solve the given problem and return the numerical result.\n"
                    "Show your work clearly and give the final answer.\n"
                    "IMPORTANT: Keep your final answer in the same units as mentioned in the problem. Don't convert units unless explicitly asked.\n"
                    "Your final answer MUST be a single number enclosed in #### markers (e.g., #### 123.45).\n\n"
                    "--- EXAMPLE ---\n"
                    'Problem: "Calculate 60 - 40."\n'
                    "Response:\n"
                    "60 minus 40 equals 20.\n"
                    "#### 20\n\n"
                    "--- YOUR TASK ---\n"
                    f'Problem: "{sub_task}"\n'
                    "Response:\n"
                )
    def _extract_math_answer(self, response: str) -> str:
        """
        extractMATHdatasetboxedanswer format
        """
        try:
            from eapae_agent_sys.data_processing.math_loader import MATH_get_predict
            return MATH_get_predict(response)
        except ImportError:
            import re
            pattern = r'\\boxed\{([^}]+)\}'
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1]
            pattern = r'####\s*(.+)'
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()
            return "ERROR_PARSING"
class CodeExecutorAgent(BaseAgent):
    """
    Accept any input（natural language or code），convert it to an equivalent、single-linePythonexpression，
    then execute it to return result。
    """
    def __init__(self, agent_config_id: str, config: Dict):
        super().__init__(agent_config_id, config)
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        instruction = state.get("instruction") or state.get("code") or state.get("sub_task") or "None"
        prompt = self._create_prompt(instruction)
        single_line_code, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=256
        )
        cleaned_code = self._clean_llm_output(single_line_code)
        actual_energy = self._calculate_actual_energy(usage)
        output_str = self._execute_code(cleaned_code)
        output_dict = {"result": output_str}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _clean_llm_output(self, code_line: str) -> str:
        """Robustly cleans the LLM output to extract a single line of executable code."""
        cleaned_code = code_line.strip()
        code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', cleaned_code, re.DOTALL)
        if code_match:
            cleaned_code = code_match.group(1).strip()
        cleaned_code = cleaned_code.replace("\n", " ").replace("`", "")
        return cleaned_code
    def _execute_code(self, single_line_code: str) -> str:
        """
        Executes a single line of Python code using eval.
        """
        try:
            safe_globals = {"__builtins__": {
                'print': print, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
                'len': len, 'str': str, 'int': int, 'float': float, 'pow': pow, 'range': range
            }}
            result = eval(single_line_code, safe_globals, {})
            return str(result)
        except Exception:
            return f"Error executing code: {traceback.format_exc()}"
    def _create_prompt(self, instruction: str) -> str:
        """
        Creates a simplified prompt to convert an instruction into a single line of Python code.
        """
        return (
            "You are a tool that converts an instruction into a single line of Python code for `eval()`.\n"
            "Your output MUST be ONLY the single line of executable code.\n"
            "The code must be self-contained (no external variables).\n\n"
            "--- EXAMPLE 1 ---\n"
            'Instruction: "Find the total cost of 8 items at $5 each and 8 items at $3 each."\n'
            "Response:\n"
            "(8 * 5) + (8 * 3)\n\n"
            "--- EXAMPLE 2 ---\n"
            'Instruction: "Take the previous result, 100, and subtract 5."\n'
            "Response:\n"
            "100 - 5\n\n"
            "--- YOUR TASK ---\n"
            'Instruction: """\n{instruction}\n"""\n'
            "Response:\n"
        ).format(instruction=instruction) 