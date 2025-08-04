from .base_agent import BaseAgent
from typing import Dict, Tuple, Any
import json
import re
try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    DDGS = None
    HAS_DUCKDUCKGO = False
class InputParserAgent(BaseAgent):
    """
    Extract and structure key numbers from initial user prompt、variables and final question。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        task_description = state.get("task_description", "")
        prompt = self._create_prompt(task_description)
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        try:
            parsed_json = json.loads(response)
            output_dict = {"parsed_data": parsed_json}
        except json.JSONDecodeError:
            output_dict = {"error": "LLM did not return valid JSON.", "raw_response": response}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, task_description: str) -> str:
        return (
            "You are a data extraction expert. Your task is to analyze the following mathematical word problem "
            "and extract key information. Please identify all key numbers, variables mentioned, and the ultimate "
            "question being asked.\n\n"
            f"Problem: \"{task_description}\"\n\n"
            "Present the output as a single, clean JSON object with three keys: 'numbers', 'variables', and 'question'. "
            "Do not include any text or explanation outside of the JSON object."
        )
class OutputFormatterAgent(BaseAgent):
    """
    Format final numerical answer to submission format required by benchmark。does not consumeLLMenergy。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        final_answer = state.get("content")
        if final_answer is None:
            last_output_dict = state.get("last_output", {})
            if isinstance(last_output_dict, dict):
                 final_answer = last_output_dict.get("result", last_output_dict.get("critique", str(last_output_dict)))
            else:
                 final_answer = str(last_output_dict)
            history = state.get("history", [])
            if history:
                final_answer = history[-1] if history else final_answer
        if isinstance(final_answer, dict):
            final_answer = final_answer.get("result", str(final_answer))
        final_answer = str(final_answer) if final_answer is not None else ""
        extracted_number = None
        hash_pattern_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', final_answer)
        if hash_pattern_match:
            extracted_number = hash_pattern_match.group(1)
        else:
            final_answer_match = re.search(r'FINAL ANSWER:\s*(-?\d+(?:\.\d+)?)', final_answer, re.IGNORECASE)
            if final_answer_match:
                extracted_number = final_answer_match.group(1)
            else:
                corrected_match = re.search(r'CORRECTED:\s*(-?\d+(?:\.\d+)?)', final_answer, re.IGNORECASE)
                if corrected_match:
                    extracted_number = corrected_match.group(1)
                else:
                    if re.match(r'^-?\d+(?:\.\d+)?$', final_answer.strip()):
                        extracted_number = final_answer.strip()
                    else:
                        numbers = re.findall(r'(-?\d+(?:\.\d+)?)', final_answer)
                        if numbers:
                            extracted_number = numbers[-1]
        if extracted_number:
            formatted_answer = f"The final answer is {extracted_number}."
        else:
            if final_answer and final_answer.strip():
                formatted_answer = f"The final answer is {final_answer.strip()}."
            else:
                formatted_answer = "The final answer could not be determined."
        output_dict = {"formatted_answer": formatted_answer}
        usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        return output_dict, {
            "usage": usage,
            "actual_energy": 0
        }
class SimplificationAgent(BaseAgent):
    """
    Rewrite a complex sentence or sub-problem into simpler、clearer form。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        text_to_simplify = state.get("task_description", "")
        if not text_to_simplify:
            return {"simplified_text": ""}, {"usage": {"prompt_tokens": 0, "completion_tokens": 0}, "actual_energy": 0}
        prompt = self._create_prompt(text_to_simplify)
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        output_dict = {"simplified_text": response}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, text: str) -> str:
        base_prompt = f"Original text: \"{text}\"\n\n"
        if self.config_level == 'medium':
            prompt = (
                "You are an expert in clear writing. Please rewrite the following sentence or sub-problem "
                "to be simpler, clearer, and more direct, without losing the original meaning. "
                "Focus on resolving ambiguities.\n\n" +
                base_prompt +
                "Provide only the rewritten text. If it's already simple, return the original text."
            )
        else:
            prompt = (
                "Please simplify the following text. " +
                base_prompt +
                "Provide only the simplified text."
            )
        return prompt
class WebSearchAgent(BaseAgent):
    """
    UseDuckDuckGosearch web to find specific constants、formulas or factual question answers。
    """
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        query = state.get("query", "No query provided.")
        prompt = self._create_prompt(query)
        if not HAS_DUCKDUCKGO:
            response = "Web search unavailable: duckduckgo_search module not installed."
            usage = {'prompt_tokens': 0, 'completion_tokens': 0}
            actual_energy = self._calculate_actual_energy(usage)
            output_dict = {"search_result": response}
            return output_dict, {
                "usage": usage,
                "actual_energy": actual_energy
            }
        try:
            num_results = 5 if self.config_level == 'high' else 2
            with DDGS() as ddgs:
                search_results = [r for r in ddgs.text(query, max_results=num_results)]
            if not search_results:
                response = "No results found."
            else:
                context = "\n".join([f"Source: {res['title']}\nSnippet: {res['body']}" for res in search_results])
                summary_prompt = self._create_summary_prompt(query, context)
                llm_response, usage = self.llm_api.call(
                    provider=self.llm_provider,
                    model=self.model_name,
                    prompt=summary_prompt,
                    max_tokens=self.max_tokens
                )
                response = llm_response
        except Exception as e:
            response = f"An error occurred during web search: {e}"
            usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        actual_energy = self._calculate_actual_energy(usage)
        output_dict = {"search_result": response}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, query: str) -> str:
        return f"Perform a web search for the following query: '{query}'"
    def _create_summary_prompt(self, query: str, context: str) -> str:
        base_prompt = (
            f"Query: \"{query}\"\n\n"
            f"Search Results Context:\n---\n{context}\n---\n\n"
        )
        if self.config_level == 'high':
            prompt = (
                "Based on the following web search results, please provide a comprehensive and synthesized answer to the query. "
                "Cite the sources of your information from the snippets provided.\n\n" +
                base_prompt +
                "Comprehensive Answer with Citations:"
            )
        else:
            prompt = (
                "Based on the following web search results, please provide a concise answer to the query.\n\n" +
                base_prompt +
                "Concise Answer:"
            )
        return prompt
class ForceOutputAgent(BaseAgent):
    """
    As a last resort, when a plan fails mid-execution, this agent takes all available
    context and synthesizes a best-effort final answer.
    """
    def __init__(self, agent_config_id: str, config: Dict):
        super().__init__(agent_config_id, config)
    def execute(self, state: Dict) -> Tuple[Dict, Dict]:
        prompt = self._create_prompt(state)
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        final_answer = "ERROR: Failed to parse LLM response."
        try:
            start_index = response.find('{')
            end_index = response.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_str = response[start_index:end_index+1]
                try:
                    parsed_json = json.loads(json_str)
                    final_answer = parsed_json.get("final_answer", "ERROR: LLM failed to provide final_answer key in JSON.")
                except json.JSONDecodeError:
                    final_answer = f"ERROR: Failed to decode extracted JSON: '{json_str}'"
            else:
                match = re.search(r'(-?\d+(\.\d+)?)', response)
                if match:
                    final_answer = match.group(0)
                else:
                    final_answer = "ERROR: Could not find JSON or a number in the response."
        except Exception as e:
            final_answer = f"ERROR: An unexpected error occurred during parsing: {e}"
        output_dict = {"result": final_answer}
        return output_dict, {
            "usage": usage,
            "actual_energy": actual_energy
        }
    def _create_prompt(self, state: Dict) -> str:
        task_description = state.get("task_description", "No task description provided.")
        history = state.get("history", [])
        reason = state.get("reason", "an unknown error")
        history_log = "\n".join([str(h) for h in history])
        if not history_log:
            history_log = "No steps were successfully completed."
        prompt = (
            f"You are a 'Final Answer' generation agent. A multi-step problem-solving process has failed mid-execution due to: **{reason}**.\n\n"
            "Your task is to provide the best possible final answer based on the original problem and the history of successfully executed steps, even though the plan was not completed.\n\n"
            "## CRITICAL INSTRUCTIONS:\n"
            "1.  **Analyze the Original Task**: Understand what was initially asked.\n"
            "2.  **Review the Execution History**: See what steps were successfully completed and what their results were.\n"
            "3.  **Synthesize and Solve**: Based on the completed steps and the original goal, deduce the final answer. You may need to perform the final calculation yourself.\n"
            "4.  **JSON Output**: Your response MUST be a single JSON object with a single key, 'final_answer', containing the final numerical answer as a string. DO NOT ADD ANY TEXT BEFORE OR AFTER THE JSON OBJECT.\n\n"
            "---\n"
            "### Original Task Description:\n"
            f"{task_description}\n\n"
            "### Execution History (what succeeded before the failure):\n"
            f"{history_log}\n\n"
            "---\n"
            "Your JSON Response (with 'final_answer' key ONLY):"
        )
        return prompt 