import json
import re
import ast
import traceback
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
class CodeCriticAgent(BaseAgent):
    """
    Specialized forMBPPcode reviewAgent。
    Evaluates code correctness、efficiency and requirement compliance。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        code_to_review = state.get("code", state.get("result", ""))
        problem_description = state.get("task_description", "")
        test_cases = state.get("test_cases", [])
        prompt = self._create_prompt(code_to_review, problem_description, test_cases)
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        try:
            critique_result = json.loads(response)
        except json.JSONDecodeError:
            try:
                import re
                json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    critique_result = json.loads(json_content)
                else:
                    raise json.JSONDecodeError("No valid JSON found", response, 0)
            except json.JSONDecodeError:
                critique_result = {
                    "is_correct": False,
                    "critique": response.strip(),
                    "suggestions": ["JSONparsing failed，please check code format"],
                    "confidence": 0.5
                }
        return critique_result, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, code: str, problem_description: str, test_cases: list) -> str:
        test_cases_str = "\n".join(test_cases) if test_cases else "No test cases provided"
        return f"""You are an expert Python code reviewer specializing in programming problems.
Review the following code solution and provide a comprehensive critique.
PROBLEM DESCRIPTION:
{problem_description}
TEST CASES:
{test_cases_str}
CODE TO REVIEW:
```python
{code}
```
EVALUATION CRITERIA:
1. **Correctness**: Does the code solve the problem correctly?
2. **Syntax**: Is the Python syntax valid?
3. **Logic**: Is the algorithmic approach sound?
4. **Test Coverage**: Will it pass the provided test cases?
5. **Efficiency**: Is the solution reasonably efficient?
6. **Code Quality**: Is the code readable and well-structured?
OUTPUT FORMAT:
Return a JSON object with exactly these fields:
{{
    "is_correct": boolean,
    "critique": "detailed analysis of the code",
    "suggestions": ["list", "of", "improvement", "suggestions"],
    "confidence": float_between_0_and_1,
    "syntax_valid": boolean,
    "logic_sound": boolean,
    "test_coverage": boolean
}}
Focus on being precise and actionable in your critique."""
class MBPPCodeSolveAgent(BaseAgent):
    """
    Specialized forMBPPtask-optimized code executionAgent。
    Supports multi-line function definitions and complexPythoncode execution。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        instruction = state.get("instruction", state.get("code", state.get("sub_task", "")))
        problem_description = state.get("task_description", "")
        test_cases = self._extract_test_cases_from_description(problem_description)
        prompt = self._create_prompt(instruction, problem_description, test_cases)
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        cleaned_code = self._clean_llm_output(response)
        syntax_valid = self._validate_syntax(cleaned_code)
        if syntax_valid:
            result = cleaned_code
        else:
            result = f"Generated code has syntax errors:\n{cleaned_code}"
        return {"result": result}, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _extract_test_cases_from_description(self, description: str) -> list:
        """Extract test cases from enhanced task description"""
        test_cases = []
        if "Test cases for reference:" in description:
            lines = description.split('\n')
            in_test_cases = False
            for line in lines:
                if "Test cases for reference:" in line:
                    in_test_cases = True
                    continue
                if in_test_cases and line.strip():
                    if line.strip().startswith('assert '):
                        test_cases.append(line.strip())
        return test_cases
    def _clean_llm_output(self, response: str) -> str:
        """CleanLLMoutput，extractPythoncode"""
        code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response.strip()
    def _validate_syntax(self, code: str) -> bool:
        """validatePythoncode syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    def _create_prompt(self, instruction: str, problem_description: str, test_cases: list) -> str:
        test_cases_str = "\n".join(test_cases) if test_cases else "No test cases provided"
        return f"""You are a Python code generator for programming problems. Generate clean, functional Python code.
PROBLEM DESCRIPTION:
{problem_description}
SPECIFIC INSTRUCTION:
{instruction}
TEST CASES (for reference - analyze these to determine the correct function name and expected behavior):
{test_cases_str}
REQUIREMENTS:
1. **Function Definition**: Write a complete function that solves the problem
2. **Correct Function Name**: Analyze the test cases to determine the exact function name expected
3. **Proper Syntax**: Ensure valid Python syntax
4. **Clear Logic**: Use clear, efficient algorithms
5. **Variable Names**: Use descriptive variable names
6. **Comments**: Add simple comments for complex logic
7. **Return Statement**: Ensure the function returns the expected result format
8. **Test Compatibility**: Make sure your function works with the provided test cases
OUTPUT FORMAT:
- Provide ONLY the Python code
- No explanations or markdown formatting
- The code should be ready to execute
- The function name must match what the test cases expect
EXAMPLE:
For "Write a function to find the maximum of two numbers":
```python
def find_max(a, b):
    return max(a, b)
```
Generate the Python code for the given instruction:""" 