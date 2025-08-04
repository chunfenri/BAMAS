from .base_agent import BaseAgent
from typing import Dict, Tuple
import json
import re
class SolutionCriticAgent(BaseAgent):
    """
    Reviews a proposed solution step and its results，identifies logical fallacies or computational errors。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        problem_description = state.get("task_description", "No problem description.")
        history = state.get("history", [])
        topology_type = state.get("topology_type", "dynamic")
        dataset_type = state.get("dataset_type", "gsm8k")
        if topology_type == "fixed":
            if history:
                last_step = history[-1]
                solution_to_review = last_step.get('result', '')
                prompt = self._create_natural_dialogue_prompt(problem_description, solution_to_review, dataset_type)
            else:
                return {"critique": "CORRECT", "reason": "No solution to review"}, {"actual_energy": 0}
        else:
            if history:
                last_step = history[-1]
                last_step_and_result = (
                    f"Step {last_step.get('step')}: Executed by {last_step.get('agent')}. "
                    f"Thought: {last_step.get('thought')}. "
                    f"Input: {last_step.get('sub_task')}. "
                    f"Result: {last_step.get('result')}."
                )
            else:
                last_step_and_result = "No previous step provided."
            prompt = self._create_prompt(problem_description, last_step_and_result, dataset_type)
        response_str, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        if topology_type == "fixed":
            output_dict = self._parse_natural_response(response_str)
        else:
            is_natural_response = self._is_natural_dialogue_response(response_str)
            if is_natural_response:
                output_dict = self._parse_natural_response(response_str)
            else:
                output_dict = None
                try:
                    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        output_dict = json.loads(json_str)
                    else:
                        output_dict = {"critique": "CORRECT"}
                except json.JSONDecodeError:
                    output_dict = {"critique": "CORRECT", "raw_response": response_str}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, problem: str, last_step_and_result: str, dataset_type: str = "gsm8k") -> str:
        is_detailed_solution = (
            len(last_step_and_result) > 300 or
            "solve this problem:" in last_step_and_result.lower() or
            "review and improve" in last_step_and_result.lower() or
            "explanation" in last_step_and_result.lower()
        )
        is_fixed_topology = (
            "solve this problem:" in last_step_and_result.lower() or
            "improve your solution" in last_step_and_result.lower() or
            len(problem) < 100 and any(op in problem for op in ['+', '-', '*', '/', '='])
        )
        if is_detailed_solution or is_fixed_topology:
            if dataset_type == "math":
                return (
                    f"You are a mathematics reviewer specializing in competition mathematics. Your job is to carefully check your colleague's solution and provide helpful feedback.\n\n"
                    f"Please review the solution step by step, think through the reasoning, and then give your conclusion.\n\n"
                    f"## Original Problem\n"
                    f'{problem}\n\n'
                    f"## Solution to Review\n"
                    f'{last_step_and_result}\n\n'
                    f"## Your Review Process\n"
                    f"Please think through this systematically:\n"
                    f"1. First, understand what the original problem is asking\n"
                    f"2. Check if the mathematical approach and reasoning are sound\n"
                    f"3. Verify algebraic manipulations, calculations, and logical steps\n"
                    f"4. Confirm the final answer is properly formatted with \\boxed{{}} notation\n"
                    f"5. Ensure the answer satisfies the problem requirements\n\n"
                    f"After your analysis, please end with one of these conclusions:\n"
                    f"- If the solution is correct: \"CONCLUSION: CORRECT\"\n"
                    f"- If there are errors: \"CONCLUSION: INCORRECT - [brief explanation of the main issue]\"\n\n"
                    f"Your review:\n"
                )
            else:
                return (
                    f"You are a mathematics reviewer working with a colleague. Your job is to carefully check their solution and provide helpful feedback.\n\n"
                    f"Please review the solution step by step, think through the reasoning, and then give your conclusion.\n\n"
                    f"## Original Problem\n"
                    f'{problem}\n\n'
                    f"## Solution to Review\n"
                    f'{last_step_and_result}\n\n'
                    f"## Your Review Process\n"
                    f"Please think through this systematically:\n"
                    f"1. First, understand what the original problem is asking\n"
                    f"2. Check if the approach taken makes sense\n"
                    f"3. Verify any calculations or reasoning steps\n"
                    f"4. Determine if the final answer is correct\n\n"
                    f"After your analysis, please end with one of these conclusions:\n"
                    f"- If the solution is correct: \"CONCLUSION: CORRECT\"\n"
                    f"- If there are errors: \"CONCLUSION: INCORRECT - [brief explanation of the main issue]\"\n\n"
                    f"Your review:\n"
                )
        else:
            return (
                f"You are a meticulous assistant acting as a critic and problem solver. Your task is to deeply analyze the following reasoning step and provide a DIRECTLY EXECUTABLE corrective action if necessary.\n\n"
                f"## Overall Problem\n"
                f'"""\n{problem}\n"""\n\n'
                f"## Last Step and Result to Review\n"
                f'"""\n{last_step_and_result}\n"""\n\n'
                f"## CRITICAL INSTRUCTIONS\n"
                f"1.  Carefully analyze if the 'Last Step and Result' is a correct and logical progression towards solving the 'Overall Problem'. **Check if the input values used in the step are consistent with previous results and the problem context. Values can come from prior steps (either via `$steps[...].result` references or by using the actual computed values) or directly from the problem statement.**\n"
                f"2.  If the step and result are absolutely correct and logical, respond ONLY with the JSON object: `{{\"critique\": \"CORRECT\"}}`\n"
                f"3.  If the step or result contains ANY error (logical, calculation, dataflow, or code error), you MUST respond with a JSON object containing **three** keys:\n"
                f"    - `\"critique\": \"INCORRECT\"`\n"
                f"    - `\"reason\": \"A detailed, step-by-step explanation of the error.\"`\n"
                f"    - `\"correction\"`: This MUST be a JSON object representing a **single, directly executable task card**. The `input` field of this card MUST contain a valid, runnable `expression` for `MathExecutorAgent` or `code` for `CodeExecutorAgent` that fixes the entire step.\n"
                f"4.  Your response MUST be ONLY the JSON object. Do not add any text before or after.\n\n"
                f"## Example 1: INCORRECT (Calculation Error)\n"
                f'{{"critique": "INCORRECT", "reason": "The profit calculation is wrong. It should be revenue minus the cost of the house AND the repairs.", "correction": {{"agent": "Executor:MathExecutorAgent", "input": {{"expression": "200000 - (80000 + 50000)"}}}}}}'
                f"\n\n"
                f"## Example 2: INCORRECT (Code Execution Error)\n"
                f'{{"critique": "INCORRECT", "reason": "The original code failed because of a NameError. The `range` function was missing. The correction adds it back.", "correction": {{"agent": "Executor:CodeExecutorAgent", "input": {{"code": "total = 0\\nfor i in range(5):\\n  total += i\\ntotal"}}}}}}'
                f"\n\n## Your Response (JSON only):\n"
            )
    def _is_natural_dialogue_response(self, response: str) -> bool:
        """Check if response is natural conversation format"""
        return "CONCLUSION:" in response.upper() or (
            len(response) > 50 and not response.strip().startswith('{')
        )
    def _parse_natural_response(self, response: str) -> Dict:
        """ParseAutoGenstyle natural conversation response，prioritize finding fixed format"""
        conclusion_match = re.search(r'CONCLUSION:\s*(CORRECT|INCORRECT)(?:\s*-\s*(.+))?', response, re.IGNORECASE)
        if conclusion_match:
            status = conclusion_match.group(1).upper()
            reason = conclusion_match.group(2) if conclusion_match.group(2) else ""
            if status == "CORRECT":
                return {
                    "critique": "CORRECT",
                    "review": response,
                    "result": response
                }
            else:
                return {
                    "critique": "INCORRECT", 
                    "reason": reason.strip() if reason else "Solution contains errors",
                    "review": response,
                    "result": response
                }
            response_lower = response.lower()
            final_answer_positive = any(phrase in response_lower for phrase in [
                "final answer is correct", "answer is correct", "final answer given is correct",
                "answer given is correct", "final result is correct", "final answer.*correct"
            ])
            final_answer_negative = any(phrase in response_lower for phrase in [
                "final answer is wrong", "answer is wrong", "final answer is incorrect",
                "answer is incorrect", "final result is wrong", "final answer.*wrong"
            ])
            if final_answer_positive and not final_answer_negative:
                return {
                    "critique": "CORRECT",
                    "reason": "Final answer is correct despite process issues",
                    "review": response,
                    "result": response
                }
            elif final_answer_negative:
                return {
                    "critique": "INCORRECT",
                    "reason": "Final answer is incorrect",
                    "review": response, 
                    "result": response
                }
            negative_indicators = ["incorrect", "wrong", "error", "mistake", "not correct", "fails","misunderstood", "misinterpretation", 
                                    "misinterpreted", "not accurate interpretation", "inaccurate", "incorrectly", "incorrectly calculated", "incorrectly solved"]
            positive_indicators = ["correct", "right", "accurate", "good", "valid", "solution is right"]
            has_negative = any(neg in response_lower for neg in negative_indicators)
            has_positive = any(pos in response_lower for pos in positive_indicators)
            missing_info = any(phrase in response_lower for phrase in [
                "no previous step", "no answer is given", "no reasoning is shown",
                "cannot verify", "cannot confirm", "not provided"
            ])
            if has_negative or missing_info:
                return {
                    "critique": "INCORRECT",
                    "reason": "Solution appears incomplete or contains issues based on review",
                    "review": response, 
                    "result": response
                }
            elif has_positive:
                return {
                    "critique": "CORRECT",
                    "review": response,
                    "result": response
                }
            else:
                return {
                    "critique": "INCORRECT",
                    "reason": "Unable to determine correctness from review",
                    "review": response, 
                    "result": response
                }
    def _create_natural_dialogue_prompt(self, problem: str, solution: str, dataset_type: str = "gsm8k") -> str:
        """Create for fixed topologyAutoGenstyle natural conversationprompt，specify output format requirements"""
        if dataset_type == "math":
            return (
                f"You are a mathematics reviewer specializing in competition mathematics. Your task is to briefly analyze the solution and give a clear conclusion.\n\n"
                f"## Problem to Solve\n"
                f'{problem}\n\n'
                f"## Solution to Review\n"
                f'{solution}\n\n'
                f"## Analysis Requirements\n"
                f"Please analyze briefly:\n"
                f"1. Check if the mathematical reasoning is sound\n"
                f"2. Verify if the calculations and algebraic manipulations are correct\n"
                f"3. Confirm if the final answer is properly formatted in \\boxed{{}} notation\n"
                f"4. CRITICAL: Check if the final answer actually answers what the problem is asking for\n"
                f"   - Read the question carefully: what specific quantity is being asked?\n"
                f"   - Make sure the final answer provides that exact quantity, not a related calculation\n\n"
                f"## IMPORTANT: Output Format\n"
                f"After your brief analysis, you MUST end with exactly one of these two lines:\n"
                f"CONCLUSION: CORRECT\n"
                f"CONCLUSION: INCORRECT\n\n"
                f"Your analysis:\n"
            )
        else:
            return (
                    f"You are a mathematics reviewer. Your task is to briefly analyze the solution and give a clear conclusion.\n\n"
                    f"## Problem to Solve\n"
                f'{problem}\n\n'
                f"## Solution to Review\n"
                f'{solution}\n\n'
                    f"## Analysis Requirements\n"
                    f"Please analyze briefly:\n"
                    f"1. Check if you understand the problem correctly\n"
                    f"2. Verify if the calculation process is correct\n"
                    f"3. Confirm if the final answer makes sense\n\n"
                    f"## IMPORTANT: Output Format\n"
                    f"After your brief analysis, you MUST end with exactly one of these two lines:\n"
                    f"CONCLUSION: CORRECT\n"
                    f"CONCLUSION: INCORRECT\n\n"
                    f"Your analysis:\n"
            )