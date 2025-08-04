import json
import re
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
class PlannerAgent(BaseAgent):
    """
    The tactical planner for the dynamic execution engine.
    At each step, it analyzes the history and decides on the next
    action, returning it as a machine-readable "Task Card".
    It is resource-aware and responsible for graceful task completion.
    """
    def _format_resources_for_prompt(self, resources: Dict[str, Any]) -> str:
        """Formats the complex resource dictionary into a readable string for the LLM."""
        lines = []
        for key, details in resources.items():
            role, level = key.split('_')
            agent_choices_str = ", ".join(details.get('agent_choices', []))
            slots = details.get('slots_available', 0)
            lines.append(f"- {role} ({level}): {slots} slot(s) available. Choices: [{agent_choices_str}]")
        return "\\n".join(lines)
    def _create_prompt(self, state: Dict) -> str:
        """Creates a robust, resource-aware prompt to guide the LLM."""
        collaboration_pattern = state.get('collaboration_pattern')
        pattern_name = collaboration_pattern.get('name') if collaboration_pattern else 'linear_chain'
        if pattern_name == 'delegate_and_gather':
            return self._create_delegate_and_gather_prompt(state)
        elif pattern_name in ['reflection_loop', 'iterative_refinement']:
            return self._create_critic_based_prompt(state, pattern_name)
        else:
            return self._create_linear_prompt(state)
    def _create_base_prompt_template(self) -> str:
        """A base template for all prompts to ensure consistency."""
        return (
            "You are an expert planner. Create a JSON list of 'Task Cards' to solve the given problem.\n\n"
            "## CORE RULES:\n"
            "1.  **JSON Output**: Your output MUST be a valid JSON list of task cards, and nothing else.\n"
            "2.  **Agent Selection**: Use the 'agent' field to specify the agent for each task using the format 'Role:ClassName' (e.g., 'Executor:MathExecutorAgent', 'Critic:SolutionCriticAgent'). Choose from the provided resource pool.\n"
            "3.  **Concrete Executor Instructions**: For any Executor Agent (`MathExecutorAgent`, `CodeExecutorAgent`), the `input` field MUST contain a single `instruction` key. The value must be a self-contained, specific command with all necessary numbers and data.\n"
            "    - **For the FIRST step**: Use actual numbers from the problem (e.g., \"Calculate 10 + (4 * 20)\")\n"
            "    - **For SUBSEQUENT steps**: Use `$steps[<index>].result` to reference previous results (e.g., \"Calculate $steps[0].result * 0.20\")\n"
            "    - **NEVER use vague references** like \"previous result\", \"total cost\", or \"the answer from before\"\n"
            "4.  **Dataflow Examples**: \n"
            "    - Correct: \"Calculate tip: $steps[0].result * 0.20\"\n"
            "    - Correct: \"Add totals: $steps[0].result + $steps[1].result\"\n"
            "    - WRONG: \"Calculate 20% of previous result\"\n"
            "    - WRONG: \"Add total cost and tip\"\n"
            "5.  **Final Step**: The plan MUST end with `{{\"agent\": \"Done\", \"input\": {{\"final_answer\": \"$steps[<index>].result\"}}}}`.\n"
            "6.  **Resource Constraint**: Try your best to stay strictly within the available resource limits. Design efficient plans that merge calculations when possible to fit the budget. The system has fallback mechanisms, but your goal is to avoid needing them.\n\n"
        )
    def _create_delegate_and_gather_prompt(self, state: Dict) -> str:
        """Creates a specialized prompt for the delegate_and_gather pattern."""
        base_prompt = self._create_base_prompt_template()
        available_resources = state.get('available_resources', {})
        resource_str = self._format_resources_for_prompt(available_resources)
        num_executors = sum(d.get('slots_available', 0) for k, d in available_resources.items() if 'Executor' in k)
        if num_executors <= 1:
            return self._create_linear_prompt(state)
        num_delegates = num_executors - 1
        if num_delegates <= 1:
            return self._create_linear_prompt(state)
        specific_instructions = (
            "## METHOD: Delegate and Gather\n"
            "**This pattern is an exception to the 'merge steps' rule.** Your primary goal is to decompose the problem into as many independent parts as possible to be run in parallel. If it is not easy to decompose, you could repeat the same entire task multiple times or only use one `Executor` agent.\n"
            "1.  **Delegate**: Create at most `{num_delegates}` independent tasks for the `Executor` agents. You can use only one `Executor` agent if you think the problem is very simple or the budget is very low.\n"
            "2.  **Aggregate**: After the delegate tasks, create one `Executor` task to aggregate their results (e.g., sum up `$steps[0].result` and `$steps[1].result`).\n"
            "3.  **Resource Budget**: Keep the total number of `Executor` tasks within `{num_executors}`. Design efficient decomposition that fits this limit.\n\n"
        ).format(num_delegates=num_delegates, num_executors=num_executors)
        return (
            base_prompt +
            specific_instructions +
            "## Problem\n"
            "{task_description}\n\n"
            "## Your Available Resources\n"
            "{resource_str}\n\n"
            "## Your Plan (JSON list only):\n"
        ).format(
            task_description=state.get('task_description'),
            resource_str=resource_str
        )
    def _create_critic_based_prompt(self, state: Dict, pattern_name: str) -> str:
        """Creates a specialized prompt for patterns that require a Critic."""
        base_prompt = self._create_base_prompt_template()
        available_resources = state.get('available_resources', {})
        resource_str = self._format_resources_for_prompt(available_resources)
        num_executors = sum(d.get('slots_available', 0) for k, d in available_resources.items() if 'Executor' in k)
        num_critics = sum(d.get('slots_available', 0) for k, d in available_resources.items() if 'Critic' in k)
        if pattern_name == 'iterative_refinement':
            methodology = (
                "## METHOD: Iterative Refinement (AutoGen-style Collaborative Solving)\n"
                "Use a three-executor collaborative approach for peer review:\n"
                "1. **Initial Solution**: First `Executor` solves the problem naturally and clearly. The instruction should be: 'Solve this problem: [full problem text]'\n"
                "2. **Peer Review**: Second `Executor` reviews and improves the first solution. The instruction should be: 'Review and improve this solution: [previous complete solution]'\n"
                "3. **Final Verification**: Third `Executor` verifies and confirms the final answer. The instruction should be: 'Verify this solution and confirm the final answer: [previous complete solution]'\n"
                "**Context Passing**: Each step should receive the COMPLETE output from previous steps, not just numerical results.\n"
                "**Resource Budget**: You have **{num_executors}** `Executor` agent(s) available. Design a 3-step plan: Solve → Review → Verify.\n"
            ).format(num_executors=num_executors)
        else:
            methodology = (
                "## METHOD: Reflection Loop\n"
                "First, solve the entire problem. Then, use the `Critic` to reflect on the final result.\n"
                "1.  **Solve**: Use up to **{num_executors}** `Executor` agent(s) to solve the problem. You SHOULD combine calculations to stay within this limit. The last of these steps should produce the final numerical answer.\n"
                "2.  **Reflect**: Immediately after the final calculation step, add a single step using the `Critic:SolutionCriticAgent` agent. The `Critic` will automatically review the result of the 'Solve' step.\n"
                "3.  **Finalize**: The plan MUST end with the `Done` agent. It should report the result from the **'Solve' step**, NOT the 'Reflect' step. Example: {{{{ 'final_answer': '$steps[0].result' }}}} if the solve step was the first task.\n"
                "**Resource Budget**: You have **{num_executors}** `Executor` agents available. Design your plan to work strictly within this limit.\n"
            ).format(num_executors=num_executors)
        return (
            base_prompt +
            methodology + "\n" +
            "## Problem\n"
            "{task_description}\n\n"
            "## Your Available Resources\n"
            "{resource_str}\n\n"
            "## Your Plan (JSON list only):\n"
        ).format(
            task_description=state.get('task_description'),
            resource_str=resource_str
        )
    def _create_linear_prompt(self, state: Dict) -> str:
        """Creates a simple, sequential prompt for linear patterns."""
        base_prompt = self._create_base_prompt_template()
        available_resources = state.get('available_resources', {})
        resource_str = self._format_resources_for_prompt(available_resources)
        num_executors = sum(d.get('slots_available', 0) for k, d in available_resources.items() if 'Executor' in k)
        constraint_instruction = (
            "## METHOD: Linear Chain\n"
            "Create a simple, minimum number of steps plan (most cases could be solved in one step or two steps). The plan could be very short (Even one step contains all the calculation). You have **{num_executors}** `Executor` agent(s) available. Your plan should be as short as possible and work within this limit.\n\n"
        ).format(num_executors=num_executors)
        return (
            base_prompt +
            constraint_instruction +
            "## Problem\n"
            "{task_description}\n\n"
            "## Your Available Resources\n"
            "{resource_str}\n\n"
            "**Resource Budget**: You have {num_executors} `Executor` agents available. Design your plan to work within this limit by combining calculations when possible.\n"
            "## Your Plan (JSON list only):\n"
        ).format(
            task_description=state.get('task_description'),
            resource_str=resource_str,
            num_executors=num_executors
        )
    def execute(self, state: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generates the next Task Card by calling the LLM and robustly parsing the response.
        """
        prompt = self._create_prompt(state)
        max_tokens_for_call = self.max_tokens
        collaboration_pattern = state.get('collaboration_pattern')
        if collaboration_pattern and collaboration_pattern.get('name') in ['delegate_and_gather', 'iterative_refinement', 'reflection_loop']:
            max_tokens_for_call = 2048
        response_text, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens_for_call
        )
        parsed_response = None
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                def escape_code_newlines(match):
                    code_content = match.group(2)
                    escaped_content = code_content.replace('\n', '\\n')
                    return f'{match.group(1)}{escaped_content}{match.group(3)}'
                json_str = re.sub(r'("code":\s*")((?:.|\n)*?)(")', escape_code_newlines, json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                json_str = re.sub(r',\s*\}', '}', json_str)
                try:
                    parsed_response = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str_fixed = re.sub(
                        r'("final_answer":\s*)"[^"]*"\s*\+\s*("(\$steps\[\d+\]\.result)")',
                        r'\1\2',
                        json_str
                    )
                    parsed_response = json.loads(json_str_fixed)
            else:
                print(f"ERROR: PlannerAgent response contained no valid JSON list. Response: {response_text}")
        except json.JSONDecodeError as e:
            print(f"ERROR: PlannerAgent failed to produce valid JSON. Error: {e}. Response: {response_text}")
        if parsed_response is None:
            parsed_response = [{"agent": "Done", "input": {"error": "Planner failed to generate a valid action."}}]
        actual_energy = self._calculate_actual_energy(usage)
        return parsed_response, {
            "usage": usage,
            "actual_energy": actual_energy
        }