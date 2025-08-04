import importlib
import json
import re
from typing import Dict, Any, Tuple
import copy
from eapae_agent_sys.utils.config_loader import ConfigLoader
from eapae_agent_sys.agents.base_agent import BaseAgent
from eapae_agent_sys.utils.evaluation import evaluate_success
class AgentFactory:
    """Dynamically creates agent instances from their IDs."""
    _agent_class_mapping = {
        "PlannerAgent": "eapae_agent_sys.agents.planner_agent.PlannerAgent",
        "MathExecutorAgent": "eapae_agent_sys.agents.executor_agents.MathExecutorAgent",
        "CodeExecutorAgent": "eapae_agent_sys.agents.executor_agents.CodeExecutorAgent",
        "MBPPCodeSolveAgent": "eapae_agent_sys.agents.mbpp_agents.MBPPCodeSolveAgent",
        "SolutionCriticAgent": "eapae_agent_sys.agents.critic_agent.SolutionCriticAgent",
        "CodeCriticAgent": "eapae_agent_sys.agents.mbpp_agents.CodeCriticAgent",
        "InputParserAgent": "eapae_agent_sys.agents.utility_agents.InputParserAgent",
        "OutputFormatterAgent": "eapae_agent_sys.agents.utility_agents.OutputFormatterAgent",
        "SimplificationAgent": "eapae_agent_sys.agents.utility_agents.SimplificationAgent",
        "WebSearchAgent": "eapae_agent_sys.agents.utility_agents.WebSearchAgent",
        "ForceOutputAgent": "eapae_agent_sys.agents.utility_agents.ForceOutputAgent",
        "PoetryAgent": "eapae_agent_sys.agents.irrelevant_agents.PoetryAgent",
        "MarketingCopyAgent": "eapae_agent_sys.agents.irrelevant_agents.MarketingCopyAgent",
        "HistoricalFactAgent": "eapae_agent_sys.agents.irrelevant_agents.HistoricalFactAgent",
        "CodeRefactorAgent": "eapae_agent_sys.agents.irrelevant_agents.CodeRefactorAgent",
    }
    @staticmethod
    def create_agent(agent_config_id: str, config: Dict[str, Any]) -> BaseAgent:
        """
        Creates an agent instance based on its configuration ID.
        """
        agent_class_name = config.get('class_name')
        if not agent_class_name:
            raise ValueError(f"No class_name found in config for agent {agent_config_id}")
        agent_class_path = AgentFactory._agent_class_mapping.get(agent_class_name)
        if not agent_class_path:
            raise ValueError(f"Unknown agent class: {agent_class_name}")
        module_path, class_name = agent_class_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import or find class {class_name} in {module_path}: {e}")
        return agent_class(agent_config_id=agent_config_id, config=config)
class ExecutionEngine:
    """
    Implements the Dynamic Instruction-Dispatch Architecture.
    This engine manages a pool of agent resources and executes a task
    by dynamically calling a PlannerAgent to get the next instruction,
    and then dispatching an available agent to perform it.
    It can now dynamically re-provision agents if a role is exhausted.
    """
    def __init__(self, config_loader: ConfigLoader, low_level_instantiator):
        """
        Initializes the execution engine.
        Args:
            config_loader: An instance of ConfigLoader holding all configurations.
            low_level_instantiator: An instance of the ILP solver to allow for dynamic re-planning.
        """
        self.config_loader = config_loader
        self.agent_factory = AgentFactory()
        self.low_level_instantiator = low_level_instantiator
        self.planner_instance = self._create_agent_instance_by_id("PlannerAgent_High")
        self.resource_pool = {}
        self.initial_pool = {}
    def _force_final_answer(self, context: dict, reason: str) -> tuple[dict, float]:
        """
        Invokes the ForceOutputAgent to generate a best-effort final answer when the plan fails.
        """
        print(f"\n[!] Plan execution failed due to: {reason}. Invoking ForceOutputAgent as a last resort...")
        force_agent = self._create_agent_instance_by_id("ForceOutputAgent_System")
        state_for_force_agent = {
            "task_description": context.get("task_description"),
            "history": context.get("history"),
            "reason": reason
        }
        final_result, metadata = force_agent.execute(state_for_force_agent)
        cost = metadata.get("actual_energy", 0)
        final_answer = final_result.get("result", "ForceOutputAgent failed to produce a result.")
        context["history"].append({
            "step": len(context.get("history", [])) + 1,
            "agent": "ForceOutputAgent",
            "thought": "Synthesizing a final answer after execution failure.",
            "result": final_answer
        })
        return context, cost
    def _create_agent_instance_by_id(self, agent_id: str) -> BaseAgent:
        """Helper to instantiate an agent by its ID from the config."""
        agent_config = self.config_loader.agents.get(agent_id)
        if not agent_config:
            if agent_id == "ForceOutputAgent_System":
                agent_config = {
                    "id": "ForceOutputAgent_System", "class_name": "ForceOutputAgent", 
                    "provider": "deepseek", "model_name": "deepseek-chat", "max_tokens": 512,
                    "cost_model": {"per_1k_prompt_tokens": 0.01, "per_1k_completion_tokens": 0.02}
                }
            else:
                raise ValueError(f"Agent config for ID '{agent_id}' not found.")
        return self.agent_factory.create_agent(agent_id, agent_config)
    def _substitute_placeholders(self, data: Any, step_outputs: list) -> Any:
        """
        Recursively substitutes placeholders like '$steps[0].result' in the input data.
        """
        if isinstance(data, dict):
            return {k: self._substitute_placeholders(v, step_outputs) for k, v in data.items()}
        if isinstance(data, list):
            return [self._substitute_placeholders(elem, step_outputs) for elem in data]
        if isinstance(data, str):
            placeholder_match = re.fullmatch(r"\$steps\[(\d+)\]\.result", data)
            if placeholder_match:
                step_index = int(placeholder_match.group(1))
                if step_index < len(step_outputs):
                    return step_outputs[step_index]
                else:
                    raise ValueError(f"Invalid step index {step_index} in placeholder. Only {len(step_outputs)} outputs available.")
            def replace_match(match):
                step_index = int(match.group(1))
                if step_index < len(step_outputs):
                    return str(step_outputs[step_index])
                else:
                    raise ValueError(f"Invalid step index {step_index} in placeholder. Only {len(step_outputs)} outputs available.")
            return re.sub(r"\$steps\[(\d+)\]\.result", replace_match, data)
        return data
    def _manage_resource_pool(self, initial_pool: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the ILP output into a live, manageable resource tracker."""
        live_pool = {}
        if not initial_pool or 'pool' not in initial_pool:
            return live_pool
        for role, levels in initial_pool['pool'].items():
            live_pool[role] = {}
            for level, details in levels.items():
                live_pool[role][level] = {
                    'slots_total': details['count'],
                    'slots_used': 0,
                    'agent_choices': details['agents'],
                    'cost_per_instance': details.get('cost_per_instance', 0)
                }
        return live_pool
    def _count_agents_in_plan(self, plan: list) -> dict:
        """Counts the required number of agents based on their class name from a plan."""
        counts = {}
        if not isinstance(plan, list):
            return counts
        for step in plan:
            agent_specifier = step.get("agent")
            if agent_specifier and agent_specifier != 'Done':
                agent_class_name = agent_specifier.split(':')[-1]
                counts[agent_class_name] = counts.get(agent_class_name, 0) + 1
        return counts
    def _count_available_agents_in_pool(self, pool: dict) -> dict:
        """Counts the available number of agents based on their class name from a resource pool."""
        counts = {}
        for role, levels in pool.items():
            for level, data in levels.items():
                for agent_class_name in data.get('agents', []):
                    counts[agent_class_name] = counts.get(agent_class_name, 0) + data.get('count', 0)
        return counts
    def _get_total_resources_for_planner(self, live_pool: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the live pool into a planner-friendly dictionary showing TOTAL slots."""
        planner_view = {}
        for role, levels in live_pool.items():
            for level, details in levels.items():
                slots_total = details.get('slots_total', details.get('count', 0))
                agent_choices = details.get('agent_choices', details.get('agents', []))
                if slots_total > 0:
                    key = f"{role}_{level}"
                    planner_view[key] = {
                        "slots_available": slots_total,
                        "agent_choices": agent_choices
                    }
        return planner_view
    def _get_available_resources_for_planner(self, live_pool: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the live pool into a planner-friendly dictionary."""
        planner_view = {}
        for role, levels in live_pool.items():
            for level, details in levels.items():
                slots_available = details['slots_total'] - details['slots_used']
                if slots_available > 0:
                    key = f"{role}_{level}"
                    planner_view[key] = {
                        "slots_available": slots_available,
                        "agent_choices": details['agent_choices']
                    }
        return planner_view
    def _calculate_min_replan_budget(self) -> float:
        """
        Calculates the minimum budget required to run a Planner and an Executor.
        This is used to decide if a full re-plan is viable.
        """
        min_planner_cost = float('inf')
        min_executor_cost = float('inf')
        avg_prompt_tokens = self.config_loader.params.get('ilp_solver', {}).get('avg_prompt_tokens', 250)
        for agent_config in self.config_loader.agents.values():
            max_completion_tokens = agent_config.get('max_tokens', 256)
            cost = (
                agent_config.get('cost_coeff_A', 0) * avg_prompt_tokens +
                agent_config.get('cost_coeff_B', 0) * max_completion_tokens
            )
            if agent_config.get('role') == 'Planner':
                if cost < min_planner_cost:
                    min_planner_cost = cost
            if agent_config.get('role') == 'Executor':
                if cost < min_executor_cost:
                    min_executor_cost = cost
        if min_planner_cost == float('inf') or min_executor_cost == float('inf'):
            return float('inf')
        return min_planner_cost + min_executor_cost
    def _check_resource_availability(self, role_to_check: str, pool: dict) -> bool:
        """Checks if any agent for a given role has available slots."""
        for role, levels in pool.items():
            if role == role_to_check:
                for level, data in levels.items():
                    if data.get('count', 0) > 0:
                        return True
        return False
    def _find_and_consume_slot(self, agent_class_name: str, live_pool: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str]:
        """Finds an available slot for a given agent class, consumes it, and returns the agent's full config."""
        agent_config = next((ac for ac in self.config_loader.agents.values() if ac['class_name'] == agent_class_name), None)
        if not agent_config:
            return None, f"Agent class '{agent_class_name}' not found in agent library."
        agent_id_prefix = agent_config['id'].split('_')[0]
        full_agent_id = f"{agent_id_prefix}_{agent_config['config_level']}"
        agent_role = agent_config.get('role')
        agent_level = agent_config.get('config_level')
        if not agent_role or not agent_level:
            return None, f"Configuration for agent class '{agent_class_name}' is missing 'role' or 'config_level'."
        if agent_role in live_pool and agent_level in live_pool[agent_role]:
            pool_slot = live_pool[agent_role][agent_level]
            if pool_slot['slots_used'] < pool_slot['slots_total']:
                pool_slot['slots_used'] += 1
                full_config = self.config_loader.agents.get(full_agent_id)
                if not full_config:
                    return None, f"Could not find full config for agent ID '{full_agent_id}'."
                full_config['cost'] = pool_slot['cost_per_instance']
                return full_config, "Slot consumed successfully."
            else:
                return None, f"Resource exhausted for role '{agent_role}' at level '{agent_level}'."
        else:
            return None, f"No slots were ever provisioned for role '{agent_role}' at level '{agent_level}'."
    def _is_team_viable(self, agent_pool: Dict[str, Any]) -> bool:
        """
        Checks if a provisioned agent pool is viable for execution.
        A viable team MUST have at least one Planner and one Executor.
        """
        if not agent_pool or agent_pool.get('status') != 'Optimal':
            return False
        roles_present = agent_pool.get('pool', {}).keys()
        if 'Planner' in roles_present and 'Executor' in roles_present:
            return True
        return False
    def _get_required_roles_from_plan(self, plan: list) -> dict:
        """
        Parses a plan to count the number of times each unique role (e.g., Planner_High) is required.
        """
        role_counts = {}
        if not isinstance(plan, list):
            return role_counts
        for step in plan:
            agent_specifier = step.get("agent")
            if not agent_specifier or agent_specifier == "Done":
                continue
            role_part = agent_specifier.split(':')[0]
            role_counts[role_part] = role_counts.get(role_part, 0) + 1
        return role_counts
    def _count_available_agents_in_pool(self, pool: dict) -> dict:
        """Counts the available number of agents based on their class name from a resource pool."""
        counts = {}
        for role, levels in pool.items():
            for level, data in levels.items():
                for agent_class_name in data.get('agents', []):
                    counts[agent_class_name] = counts.get(agent_class_name, 0) + data.get('count', 0)
        return counts
    def _get_total_resources_for_planner(self, live_pool: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the live pool into a planner-friendly dictionary showing TOTAL slots."""
        planner_view = {}
        for role, levels in live_pool.items():
            for level, details in levels.items():
                slots_total = details.get('slots_total', details.get('count', 0))
                agent_choices = details.get('agent_choices', details.get('agents', []))
                if slots_total > 0:
                    key = f"{role}_{level}"
                    planner_view[key] = {
                        "slots_available": slots_total,
                        "agent_choices": agent_choices
                    }
        return planner_view
    def _get_available_resources_for_planner(self, live_pool: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the live pool into a planner-friendly dictionary."""
        planner_view = {}
        for role, levels in live_pool.items():
            for level, details in levels.items():
                slots_available = details['slots_total'] - details['slots_used']
                if slots_available > 0:
                    key = f"{role}_{level}"
                    planner_view[key] = {
                        "slots_available": slots_available,
                        "agent_choices": details['agent_choices']
                    }
        return planner_view
    def _calculate_min_replan_budget(self) -> float:
        """
        Calculates the minimum budget required to run a Planner and an Executor.
        This is used to decide if a full re-plan is viable.
        """
        min_planner_cost = float('inf')
        min_executor_cost = float('inf')
        avg_prompt_tokens = self.config_loader.params.get('ilp_solver', {}).get('avg_prompt_tokens', 250)
        for agent_config in self.config_loader.agents.values():
            max_completion_tokens = agent_config.get('max_tokens', 256)
            cost = (
                agent_config.get('cost_coeff_A', 0) * avg_prompt_tokens +
                agent_config.get('cost_coeff_B', 0) * max_completion_tokens
            )
            if agent_config.get('role') == 'Planner':
                if cost < min_planner_cost:
                    min_planner_cost = cost
            if agent_config.get('role') == 'Executor':
                if cost < min_executor_cost:
                    min_executor_cost = cost
        if min_planner_cost == float('inf') or min_executor_cost == float('inf'):
            return float('inf')
        return min_planner_cost + min_executor_cost
    def _check_resource_availability(self, role_to_check: str, pool: dict) -> bool:
        """Checks if any agent for a given role has available slots."""
        for role, levels in pool.items():
            if role == role_to_check:
                for level, data in levels.items():
                    if data.get('count', 0) > 0:
                        return True
        return False
    def _find_and_consume_slot(self, agent_class_name: str, live_pool: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str]:
        """Finds an available slot for a given agent class, consumes it, and returns the agent's full config."""
        agent_config = next((ac for ac in self.config_loader.agents.values() if ac['class_name'] == agent_class_name), None)
        if not agent_config:
            return None, f"Agent class '{agent_class_name}' not found in agent library."
        agent_id_prefix = agent_config['id'].split('_')[0]
        full_agent_id = f"{agent_id_prefix}_{agent_config['config_level']}"
        agent_role = agent_config.get('role')
        agent_level = agent_config.get('config_level')
        if not agent_role or not agent_level:
            return None, f"Configuration for agent class '{agent_class_name}' is missing 'role' or 'config_level'."
        if agent_role in live_pool and agent_level in live_pool[agent_role]:
            pool_slot = live_pool[agent_role][agent_level]
            if pool_slot['slots_used'] < pool_slot['slots_total']:
                pool_slot['slots_used'] += 1
                full_config = self.config_loader.agents.get(full_agent_id)
                if not full_config:
                    return None, f"Could not find full config for agent ID '{full_agent_id}'."
                full_config['cost'] = pool_slot['cost_per_instance']
                return full_config, "Slot consumed successfully."
            else:
                return None, f"Resource exhausted for role '{agent_role}' at level '{agent_level}'."
        else:
            return None, f"No slots were ever provisioned for role '{agent_role}' at level '{agent_level}'."
    def _is_team_viable(self, agent_pool: Dict[str, Any]) -> bool:
        """
        Checks if a provisioned agent pool is viable for execution.
        A viable team MUST have at least one Planner and one Executor.
        """
        if not agent_pool or agent_pool.get('status') != 'Optimal':
            return False
        roles_present = agent_pool.get('pool', {}).keys()
        if 'Planner' in roles_present and 'Executor' in roles_present:
            return True
        return False
    def _get_required_roles_from_plan(self, plan: list) -> dict:
        """
        Parses a plan to count the number of times each unique role (e.g., Planner_High) is required.
        """
        role_counts = {}
        if not isinstance(plan, list):
            return role_counts
        for step in plan:
            agent_specifier = step.get("agent")
            if not agent_specifier or agent_specifier == "Done":
                continue
            role_part = agent_specifier.split(':')[0]
            role_counts[role_part] = role_counts.get(role_part, 0) + 1
        return role_counts
    def _create_sanitized_planner_state(self, original_context: Dict, planning_reason: str, total_resource_view: Dict) -> Dict:
        """
        Creates a sanitized state dictionary for the planner, abstracting away execution details.
        Args:
            original_context: The current internal state of the engine.
            planning_reason: A string explaining why planning is being triggered.
            total_resource_view: A dict representing all *initially* available agents for the planner.
        Returns:
            A state dictionary suitable for the PlannerAgent.
        """
        return {
            "task_description": original_context.get("task_description"),
            "available_resources": total_resource_view,
            "history": copy.deepcopy(original_context.get("history", [])),
            "reason_for_plan": planning_reason
        }
    def execute(self, agent_pool: Dict[str, Any], initial_budget: float, task_description: str, dataset_type: str = "gsm8k") -> Tuple[Dict[str, Any], float]:
        """
        Main execution loop.
        Args:
            agent_pool: The ILP solver's output, defining the team and costs.
            initial_budget: The total energy budget for the task.
            task_description: The user's problem description.
            dataset_type: The type of dataset being processed (e.g., "gsm8k", "math").
        Returns:
            A tuple containing the final context (with history) and the total cost.
        """
        if not self._is_team_viable(agent_pool):
            return self._force_final_answer(
                context={"task_description": task_description, "history": []},
                reason="The provisioned team was not viable (missing Planner or Executor)."
            )
        live_pool = self._manage_resource_pool(agent_pool)
        total_resources_for_planner = self._get_total_resources_for_planner(live_pool)
        total_cost = agent_pool.get('total_cost', 0.0)
        remaining_budget = initial_budget - total_cost
        current_context = {
            "task_description": task_description,
            "history": [],
            "live_pool": live_pool,
            "remaining_budget": remaining_budget
        }
        try:
            planner_config, msg = self._find_and_consume_slot('PlannerAgent', live_pool)
            if not planner_config:
                raise ValueError(f"Could not secure a Planner agent: {msg}")
            planner_instance = self._create_agent_instance(planner_config)
            total_cost += planner_config.get('cost', 0)
            initial_planning_state = self._create_sanitized_planner_state(
                original_context=current_context,
                planning_reason="Initial plan generation.",
                total_resource_view=total_resources_for_planner
            )
            plan, plan_metadata = planner_instance.execute(initial_planning_state)
            plan_cost = plan_metadata.get("actual_energy", 0)
            total_cost += plan_cost
            current_context['history'].append(f"System Notice: Initial plan created at a cost of {plan_cost:.4f}. Total cost is now {total_cost:.4f}.")
            if not plan:
                raise ValueError("Planner returned an empty plan.")
        except Exception as e:
            return self._force_final_answer(current_context, f"Failed during initial planning: {e}")
        step_index = 0
        while step_index < len(plan):
            if total_cost > initial_budget:
                return self._force_final_answer(current_context, f"Execution halted: Budget exceeded. Total cost {total_cost:.4f} > Budget {initial_budget:.4f}")
            try:
                step = plan[step_index]
                agent_specifier = step.get("agent")
                if not agent_specifier or agent_specifier == "Done":
                    break 
                agent_class_name = agent_specifier.split(':')[-1]
                agent_input = step.get("input", {})
                history_results = [h.get('result') for h in current_context['history'] if isinstance(h, dict)]
                resolved_input = self._substitute_placeholders(agent_input, history_results)
                agent_config, message = self._find_and_consume_slot(agent_class_name, live_pool)
                if not agent_config:
                    print(f"[*] Failed to get agent '{agent_class_name}': {message}. Attempting to re-plan...")
                    sanitized_context = self._create_sanitized_planner_state(
                        original_context=current_context,
                        planning_reason=f"Failed to acquire agent '{agent_class_name}'. Reason: {message}",
                        total_resource_view=total_resources_for_planner
                    )
                    new_plan, replan_metadata = planner_instance.execute(sanitized_context)
                    replan_cost = replan_metadata.get("actual_energy", 0)
                    remaining_budget -= replan_cost
                    total_cost += replan_cost
                    current_context['history'].append(f"System Notice: Re-planning triggered at a cost of {replan_cost:.4f}. New total cost: {total_cost:.4f}.")
                    if not new_plan:
                        raise ValueError("Re-planning resulted in an empty plan.")
                    plan = plan[:step_index] + new_plan
                    continue
                agent_instance = self._create_agent_instance(agent_config)
                agent_cost = agent_config.get('cost', 0)
                total_cost += agent_cost
                if isinstance(resolved_input, dict):
                    resolved_input['dataset_type'] = dataset_type
                result, metadata = agent_instance.execute(resolved_input)
                step_cost = metadata.get("actual_energy", 0)
                total_cost += step_cost
                current_context['history'].append({
                    "step": len(current_context['history']) + 1,
                    "agent": agent_specifier,
                    "thought": step.get("thought", ""),
                    "input": resolved_input,
                    "result": result,
                    "cost": agent_cost + step_cost
                })
                print(f"[*] Step {step_index + 1} successful. Cost: {agent_cost + step_cost:.4f}. Total cost: {total_cost:.4f}")
            except Exception as e:
                error_message = f"Execution failed at step {step_index + 1} ('{agent_specifier}'): {e}"
                current_context['history'].append(f"Error: {error_message}")
                try:
                    sanitized_context = self._create_sanitized_planner_state(
                        original_context=current_context,
                        planning_reason=error_message,
                        total_resource_view=total_resources_for_planner
                    )
                    new_plan, replan_metadata = planner_instance.execute(sanitized_context)
                    replan_cost = replan_metadata.get("actual_energy", 0)
                    total_cost += replan_cost
                    if new_plan:
                        plan = plan[:step_index] + new_plan
                        continue
                    else:
                        raise ValueError("Re-planning failed, returned an empty plan.")
                except Exception as replan_e:
                    return self._force_final_answer(current_context, f"Re-planning failed after execution error: {replan_e}")
            step_index += 1
        final_answer_step = plan[-1] if plan else {}
        final_answer_placeholder = final_answer_step.get("input", {}).get("final_answer")
        final_context = current_context
        if final_answer_placeholder:
             history_results = [h.get('result') for h in final_context['history'] if isinstance(h, dict)]
             final_answer = self._substitute_placeholders(final_answer_placeholder, history_results)
             final_context['final_answer'] = final_answer
        else:
             final_context['final_answer'] = "No final answer designated in plan."
        return final_context, total_cost
    def _execute_step(self, agent_role_specifier: str, agent_input: Dict, resource_queues: Dict, context: Dict, remaining_budget: float, dataset_type: str = "gsm8k") -> Tuple[Dict, Dict, Dict, float, bool]:
        """
        A helper function to execute a single step, handling resource allocation and cost calculation.
        It now dispatches based on a 'ROLE:CLASS_NAME' specifier.
        Returns: output, cost_info, agent_node, actual_cost, success
        """
        if not agent_role_specifier or ':' not in agent_role_specifier:
            return {"error": f"Invalid agent specifier format: '{agent_role_specifier}'. Expected 'ROLE:CLASS_NAME'."}, {}, {}, 0.0, False
        agent_role, agent_class_name = agent_role_specifier.split(':', 1)
        if not resource_queues.get(agent_role):
            return {"error": f"Resource exhausted for role '{agent_role}'"}, {}, {}, 0.0, False
        agent_node = None
        agent_to_pop_index = -1
        queue = resource_queues[agent_role]
        for i, node in enumerate(queue):
            if node.get('class_name') == agent_class_name:
                agent_to_pop_index = i
                break
        if agent_to_pop_index == -1:
            error_msg = f"Resource exhausted for role '{agent_role}' with required class '{agent_class_name}'"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}, {}, {}, 0.0, False
        agent_node = resource_queues[agent_role].pop(agent_to_pop_index)
        agent_instance = self.agent_factory.create_agent(agent_node['agent_id'], agent_node)
        agent_state = context.copy()
        if agent_class_name == 'MathExecutorAgent' and 'expression' in agent_input:
            agent_state['sub_task'] = agent_input['expression']
        else:
            agent_state.update(agent_input)
        agent_state['dataset_type'] = dataset_type
        output, cost_info = agent_instance.execute(agent_state)
        actual_cost = cost_info.get('actual_energy', 0)
        return output, cost_info, agent_node, actual_cost, True
    def _create_fallback_agent_instance(self, agent_class_name: str) -> BaseAgent | None:
        """
        Creates a fallback agent instance when resource pool is exhausted.
        This bypasses resource constraints and directly creates an agent from config.
        """
        for agent_id, agent_config in self.config_loader.agents.items():
            if agent_config['class_name'] == agent_class_name:
                return self._create_agent_instance(agent_config)
        return None
    def execute_hybrid(self, agent_pool: dict, initial_budget: float, task_description: str, collaboration_pattern: dict = None, dataset_type: str = "gsm8k"):
        """
        New hybrid execution entry：select execution mode based on topology type
        """
        if not collaboration_pattern:
            return self._execute_with_planner(agent_pool, initial_budget, task_description, collaboration_pattern, dataset_type)
        topology_type = collaboration_pattern.get('topology_type', 'dynamic')
        execution_template = collaboration_pattern.get('execution_template', 'planner_driven')
        if topology_type == 'fixed':
            return self._execute_with_template(agent_pool, initial_budget, task_description, execution_template, dataset_type)
        else:
            return self._execute_with_planner(agent_pool, initial_budget, task_description, collaboration_pattern, dataset_type)
    def _execute_with_template(self, agent_pool: dict, initial_budget: float, task_description: str, template_name: str, dataset_type: str):
        """
        Execute fixed topology using predefined templates
        """
        if template_name == 'linear_chain':
            return self._execute_linear_template(agent_pool, initial_budget, task_description, dataset_type)
        elif template_name == 'star_dispatch':
            return self._execute_star_template(agent_pool, initial_budget, task_description, dataset_type)
        elif template_name == 'feedback_loop':
            return self._execute_feedback_template(agent_pool, initial_budget, task_description, dataset_type)
        else:
            raise ValueError(f"Unknown execution template: {template_name}")
    def _execute_linear_template(self, agent_pool: dict, initial_budget: float, task_description: str, dataset_type: str):
        """
        Linear topology template：ReferenceAutoGenlogic，sequentially execute multipleexecutor，each can see original problem and previous conversation history
        """
        print("\n" + "="*20 + " Linear Template Execution Started " + "="*20)
        current_agent_pool = copy.deepcopy(agent_pool['pool'])
        execution_history = []
        total_cost = 0
        try:
            executor_count = 0
            for role, levels in current_agent_pool.items():
                if role == 'Executor':
                    for level, data in levels.items():
                        executor_count += data.get('count', 0)
            if executor_count == 0:
                return {"history": ["Error: No Executor agents available"]}, 0
            is_simple_task = (
                len(task_description) < 50 and 
                any(op in task_description for op in ['+', '-', '*', '/', '=']) and
                any(char.isdigit() for char in task_description)
            )
            if is_simple_task:
                effective_executor_count = 1
            else:
                effective_executor_count = min(executor_count, 3)
            conversation_context = []
            reserved_high_for_final = False
            if effective_executor_count > 1:
                high_count = current_agent_pool.get('Executor', {}).get('High', {}).get('count', 0)
                low_count = current_agent_pool.get('Executor', {}).get('Low', {}).get('count', 0)
                if high_count >= 1 and low_count >= 1:
                    current_agent_pool['Executor']['High']['count'] -= 1
                    reserved_high_for_final = True
            for i in range(effective_executor_count):
                if total_cost >= initial_budget:
                    break
                if i == effective_executor_count - 1 and reserved_high_for_final:
                    if dataset_type == "mbpp":
                        executor_instance = self._instantiate_agent_by_class_and_level('MBPPCodeSolveAgent', 'High')
                    else:
                        executor_instance = self._instantiate_agent_by_class_and_level('MathExecutorAgent', 'High')
                else:
                    executor_instance = self._get_agent_instance_from_pool('Executor', current_agent_pool)
                if not executor_instance:
                    error_msg = f"Failed to get executor instance for step {i+1}"
                    execution_history.append({"error": error_msg})
                    break
                self._consume_resource('Executor', executor_instance.config['config_level'], current_agent_pool)
                if i == 0:
                    agent_input = {
                        "instruction": f"Please solve this problem step by step: {task_description}",
                        "task_description": task_description,
                        "dataset_type": dataset_type
                    }
                else:
                    conversation_summary = self._build_conversation_summary(conversation_context)
                    if i == 1:
                        agent_input = {
                            "instruction": f"""Previous conversation:
{conversation_summary}
The previous executor provided this solution to the problem "{task_description}":
{conversation_context[-1]['result']}
Please review this solution and provide an improved, more accurate solution to the original problem: {task_description}""",
                            "task_description": task_description,
                            "dataset_type": dataset_type
                        }
                    else:
                        agent_input = {
                            "instruction": f"""Previous conversation:
{conversation_summary}
Multiple executors have worked on the problem "{task_description}". 
Please carefully review the previous solutions and provide the final, verified answer to: {task_description}""",
                            "task_description": task_description,
                            "dataset_type": dataset_type
                        }
                result, metadata = executor_instance.execute(agent_input)
                step_cost = metadata.get("actual_energy", 0)
                total_cost += step_cost
                step_info = {
                    "step": i + 1,
                    "agent": f"Executor_{executor_instance.config['config_level']}",
                    "input": agent_input,
                    "result": result,
                    "cost": step_cost
                }
                execution_history.append(step_info)
                conversation_context.append({
                    "agent": f"Executor_{i+1}",
                    "result": result
                })
            print("\n" + "="*20 + " Linear Template Execution Finished " + "="*20)
            return {"history": execution_history}, total_cost
        except Exception as e:
            error_msg = f"Linear template execution failed: {e}"
            execution_history.append({"error": error_msg})
            return {"history": execution_history}, total_cost
    def _build_conversation_summary(self, conversation_context: list) -> str:
        """Build conversation history summary，for subsequentexecutorcontext"""
        if not conversation_context:
            return "No previous conversation."
        summary_parts = []
        for ctx in conversation_context:
            agent_name = ctx['agent']
            result = ctx['result']
            if isinstance(result, dict) and 'result' in result:
                result_text = str(result['result'])
            else:
                result_text = str(result)
            if len(result_text) > 150:
                result_text = result_text[:150] + "..."
            summary_parts.append(f"[{agent_name}]: {result_text}")
        return "\n".join(summary_parts)
    def _execute_star_template(self, agent_pool: dict, initial_budget: float, task_description: str, dataset_type: str):
        """
        Star topology template：decompose task to multipleexecutorparallel processing，then aggregate results
        """
        print("\n" + "="*20 + " Star Template Execution Started " + "="*20)
        current_agent_pool = copy.deepcopy(agent_pool['pool'])
        execution_history = []
        total_cost = 0
        try:
            executor_count = 0
            for role, levels in current_agent_pool.items():
                if role == 'Executor':
                    for level, data in levels.items():
                        executor_count += data.get('count', 0)
            if executor_count < 2:
                return self._execute_linear_template(agent_pool, initial_budget, task_description, dataset_type)
            subtasks = self._decompose_task_for_star(task_description, executor_count)
            subtask_results = []
            for i, subtask in enumerate(subtasks):
                if total_cost >= initial_budget:
                    break
                executor_instance = self._get_agent_instance_from_pool('Executor', current_agent_pool)
                self._consume_resource('Executor', executor_instance.config['config_level'], current_agent_pool)
                agent_input = {"instruction": subtask, "dataset_type": dataset_type}
                result, metadata = executor_instance.execute(agent_input)
                step_cost = metadata.get("actual_energy", 0)
                total_cost += step_cost
                result_with_metadata = {
                    "result": result,
                    "agent_level": executor_instance.config['config_level'],
                    "step_index": i,
                    "cost": step_cost
                }
                subtask_results.append(result_with_metadata)
                execution_history.append({
                    "step": i + 1,
                    "agent": f"Executor_{executor_instance.config['config_level']}",
                    "input": agent_input,
                    "result": result,
                    "cost": step_cost
                })
            if subtask_results and total_cost < initial_budget:
                best_result = self._select_best_result_with_strategy(subtask_results, dataset_type)
                aggregation_step = {
                    "step": len(execution_history) + 1,
                    "agent": "Aggregator_Smart",
                    "input": {"strategy": "majority_vote_with_energy_priority"},
                    "result": best_result["result"],
                    "cost": 0,
                    "decision_reason": best_result.get("decision_reason", "Smart aggregation applied")
                }
                execution_history.append(aggregation_step)
            print("\n" + "="*20 + " Star Template Execution Finished " + "="*20)
            return {"history": execution_history}, total_cost
        except Exception as e:
            error_msg = f"Star template execution failed: {e}"
            execution_history.append({"error": error_msg})
            return {"history": execution_history}, total_cost
    def _execute_feedback_template(self, agent_pool: dict, initial_budget: float, task_description: str, dataset_type: str):
        """
        Feedback topology template：executorexecute，criticevaluate，decide whether to re-execute based on feedback
        """
        print("\n" + "="*20 + " Feedback Template Execution Started " + "="*20)
        current_agent_pool = copy.deepcopy(agent_pool['pool'])
        execution_history = []
        total_cost = 0
        max_iterations = 2
        try:
            for iteration in range(max_iterations):
                if total_cost >= initial_budget:
                    break
                executor_instance = self._get_agent_instance_from_pool('Executor', current_agent_pool)
                if not executor_instance:
                    if iteration > 0:
                        prev_step = execution_history[-1] if execution_history else None
                        if prev_step and prev_step.get('agent', '').startswith('Critic'):
                            prev_critique = prev_step.get('result', {})
                            if isinstance(prev_critique, dict) and prev_critique.get("critique") == "INCORRECT":
                                if dataset_type == "mbpp":
                                    executor_instance = self._create_fallback_agent_instance('MBPPCodeSolveAgent')
                                else:
                                    executor_instance = self._create_fallback_agent_instance('MathExecutorAgent')
                                if executor_instance:
                                    break
                                else:
                                    print("[!] Fallback Executor creation failed - terminating")
                                    break
                            else:
                                break
                        else:
                            break
                    else:
                        break
                self._consume_resource('Executor', executor_instance.config['config_level'], current_agent_pool)
                if iteration == 0:
                    agent_input = {"instruction": f"Solve this problem: {task_description}", "dataset_type": dataset_type}
                else:
                    prev_feedback = execution_history[-1].get('result', '')
                    agent_input = {"instruction": f"Improve your solution to '{task_description}' based on this feedback: {prev_feedback}", "dataset_type": dataset_type}
                result, metadata = executor_instance.execute(agent_input)
                step_cost = metadata.get("actual_energy", 0)
                total_cost += step_cost
                execution_history.append({
                    "step": len(execution_history) + 1,
                    "agent": f"Executor_{executor_instance.config['config_level']}",
                    "input": agent_input,
                    "result": result,
                    "cost": step_cost
                })
                if total_cost < initial_budget:
                    critic_instance = self._get_agent_instance_from_pool('Critic', current_agent_pool)
                    if critic_instance:
                        self._consume_resource('Critic', critic_instance.config['config_level'], current_agent_pool)
                        simplified_history = [{
                            "step": 1,
                            "agent": f"Executor_{executor_instance.config['config_level']}",
                            "thought": "Solving the problem",
                            "sub_task": agent_input.get("instruction", ""),
                            "result": result
                        }]
                        if dataset_type == "mbpp":
                            code_content = ""
                            if isinstance(result, dict) and 'result' in result:
                                code_content = result['result']
                            elif isinstance(result, str):
                                code_content = result
                            else:
                                code_content = str(result)
                            critic_input = {
                                "task_description": task_description,
                                "code": code_content,
                                "result": code_content,
                                "history": simplified_history,
                                "topology_type": "fixed",
                                "dataset_type": dataset_type
                            }
                        else:
                            critic_input = {
                                "task_description": task_description,
                                "history": simplified_history,
                                "topology_type": "fixed",
                                "dataset_type": dataset_type
                            }
                        critique, critic_metadata = critic_instance.execute(critic_input)
                        critic_cost = critic_metadata.get("actual_energy", 0)
                        total_cost += critic_cost
                        execution_history.append({
                            "step": len(execution_history) + 1,
                            "agent": f"Critic_{critic_instance.config['config_level']}",
                            "input": critic_input,
                            "result": critique,
                            "cost": critic_cost
                        })
                        if isinstance(critique, dict) and critique.get("critique") == "CORRECT":
                            break
                        elif iteration == max_iterations - 1:
                            break
                    else:
                        break
                else:
                    break
            final_executor_result = None
            critic_approved = False
            for step in reversed(execution_history):
                if step.get('agent', '').startswith('Critic'):
                    critic_result = step.get('result', {})
                    if isinstance(critic_result, dict) and critic_result.get('critique') == 'CORRECT':
                        critic_approved = True
                        break
            for step in reversed(execution_history):
                if step.get('agent', '').startswith('Executor'):
                    executor_result = step.get('result', {})
                    if dataset_type == "mbpp":
                        if isinstance(executor_result, dict) and 'result' in executor_result:
                            final_executor_result = executor_result['result']
                        elif isinstance(executor_result, str):
                            final_executor_result = executor_result
                        else:
                            final_executor_result = str(executor_result)
                        break
                    if isinstance(executor_result, dict):
                        if 'numerical_answer' in executor_result and executor_result['numerical_answer'] != 'ERROR_PARSING':
                            final_executor_result = executor_result['numerical_answer']
                        elif 'result' in executor_result:
                            content = executor_result['result']
                            if isinstance(content, str):
                                extracted_answer = self._smart_extract_answer_from_reasoning(content, critic_approved)
                                if extracted_answer:
                                    final_executor_result = extracted_answer
                                else:
                                    final_executor_result = str(executor_result)
                            else:
                                final_executor_result = str(executor_result)
                        else:
                            final_executor_result = str(executor_result)
                    else:
                        final_executor_result = str(executor_result)
                    break
            if final_executor_result:
                execution_history.append({
                    "step": len(execution_history) + 1,
                    "agent": "FinalAnswer",
                    "result": final_executor_result
                })
            else:
                execution_history.append({
                    "step": len(execution_history) + 1,
                    "agent": "FinalAnswer", 
                    "result": "Unable to extract final answer",
                    "note": "Answer extraction failed despite valid reasoning"
                })
            print("\n" + "=" * 20 + " Feedback Template Execution Finished " + "=" * 20)
            return {"history": execution_history}, total_cost
        except Exception as e:
            error_msg = f"Feedback template execution failed: {e}"
            execution_history.append({"error": error_msg})
            return {"history": execution_history}, total_cost
    def _decompose_task_for_star(self, task_description: str, num_subtasks: int):
        """
        Decompose task for star topology。simple rule-based decomposition。
        """
        if "calculate" in task_description.lower() or "compute" in task_description.lower():
            subtasks = []
            if num_subtasks >= 2:
                subtasks.append(f"Extract the numbers and key information from: {task_description}")
                subtasks.append(f"Determine the mathematical operations needed for: {task_description}")
                if num_subtasks >= 3:
                    subtasks.append(f"Perform the calculation for: {task_description}")
                if num_subtasks >= 4:
                    subtasks.append(f"Verify the calculation approach for: {task_description}")
        else:
            subtasks = [f"Solve part of this problem: {task_description}" for _ in range(min(num_subtasks, 3))]
        return subtasks[:num_subtasks]
    def _select_best_result_with_strategy(self, subtask_results, dataset_type):
        """
        Use intelligent aggregation strategy to select best result：
        1. Prioritize majority answer
        2. If answer counts are equal，selectHighlevelAgentwith more answers
        3. If energy levels are the same，select the first one
        """
        if not subtask_results:
            return {"result": "No results to aggregate", "decision_reason": "Empty input"}
        if len(subtask_results) == 1:
            return {
                "result": subtask_results[0]["result"], 
                "decision_reason": "Single result available"
            }
        answer_groups = {}
        for item in subtask_results:
            result = item["result"]
            if dataset_type == "math":
                numerical_answer = self._extract_math_answer_from_result(result)
            else:
                numerical_answer = self._extract_numerical_answer_from_result(result)
            print('numerical_answer', numerical_answer)
            print('dataset_type', dataset_type)
            if numerical_answer not in answer_groups:
                answer_groups[numerical_answer] = []
            answer_groups[numerical_answer].append(item)
        for answer, items in answer_groups.items():
            levels = [item["agent_level"] for item in items]
            print(f"  Answer {answer}: {len(items)} votes from {levels}")
        if len(answer_groups) > 1:
            max_votes = max(len(items) for items in answer_groups.values())
            majority_answers = [answer for answer, items in answer_groups.items() 
                              if len(items) == max_votes]
            if len(majority_answers) == 1:
                winning_answer = majority_answers[0]
                winning_items = answer_groups[winning_answer]
                return {
                    "result": winning_items[0]["result"],
                    "decision_reason": f"Majority vote: {len(winning_items)}/{len(subtask_results)} votes for answer {winning_answer}"
                }
        if len(answer_groups) > 1:
            answer_scores = {}
            for answer, items in answer_groups.items():
                high_count = sum(1 for item in items if item["agent_level"] == "High")
                low_count = len(items) - high_count
                total_votes = len(items)
                answer_scores[answer] = {
                    "total_votes": total_votes,
                    "high_count": high_count,
                    "low_count": low_count,
                    "items": items
                }
            sorted_answers = sorted(answer_scores.items(), 
                                  key=lambda x: (x[1]["total_votes"], x[1]["high_count"]), 
                                  reverse=True)
            best_answer, best_score = sorted_answers[0]
            if len(sorted_answers) > 1:
                second_answer, second_score = sorted_answers[1]
                if (best_score["total_votes"] == second_score["total_votes"] and 
                    best_score["high_count"] > second_score["high_count"]):
                    return {
                        "result": best_score["items"][0]["result"],
                        "decision_reason": f"Energy level priority: Answer {best_answer} has {best_score['high_count']} High-level vs {second_score['high_count']} High-level for answer {second_answer}"
                    }
        first_item = min(subtask_results, key=lambda x: x["step_index"])
        if dataset_type == "math":
            first_answer = self._extract_math_answer_from_result(first_item["result"])
        else:
            first_answer = self._extract_numerical_answer_from_result(first_item["result"])
        return {
            "result": first_item["result"],
            "decision_reason": f"Time priority: Answer {first_answer} from first executor (step {first_item['step_index'] + 1})"
        }
    def _extract_numerical_answer_from_result(self, result):
        """
        FromExecutorextract numerical answer from results，for aggregation strategy comparison
        """
        if isinstance(result, dict):
            if 'numerical_answer' in result:
                answer = result['numerical_answer']
                if answer and answer != 'ERROR_PARSING':
                    return str(answer).strip()
            if 'result' in result:
                content = result['result']
                if isinstance(content, str):
                    import re
                    pattern = r'####\s*(\d+(?:\.\d+)?)'
                    match = re.search(pattern, content)
                    if match:
                        return match.group(1).strip()
                    conclusion_patterns = [
                        r'(?:final answer|answer|result|total|sum).*?(?:is|=|equals?)\s*(\d+(?:\.\d+)?)',
                        r'(?:has|have)\s*(?:a\s+)?total\s+of\s*(\d+(?:\.\d+)?)',
                        r'total\s*=\s*(\d+(?:\.\d+)?)',
                        r'(\d+(?:\.\d+)?)\s*(?:animals|items|objects|units|total|in total)',
                        r'(?:the answer is|total|there are|Total|sum)\s*[:：]?\s*(\d+(?:\.\d+)?)',
                        r'(\d+(?:\.\d+)?)\s*(?:pieces|units|heads|entries|pieces|units|items)',
                    ]
                    for pattern in conclusion_patterns:
                        match = re.search(pattern, content.lower())
                        if match:
                            return match.group(1).strip()
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        return numbers[-1].strip()
                return str(content).strip()
        elif isinstance(result, str):
            import re
            pattern = r'####\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, result)
            if match:
                return match.group(1).strip()
            numbers = re.findall(r'\d+(?:\.\d+)?', result)
            if numbers:
                return numbers[-1].strip()
        return str(result).strip()
    def _smart_extract_answer_from_reasoning(self, content: str, critic_approved: bool = False) -> str:
        """
        Intelligently extract numerical answer from reasoning content，even without standard#### format
        Especially suitable for complete reasoning but non-standard format
        """
        import re
        if not isinstance(content, str):
            return None
        pattern = r'####\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
        conclusion_patterns = [
            r'(?:final answer|answer|total|result).*?(?:is|=|equals?)\s*(\d+(?:\.\d+)?)',
            r'(?:has|have)\s*(?:a\s+)?total\s+of\s*(\d+(?:\.\d+)?)',
            r'total\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:animals|items|objects|units|total)',
            r'(?:total capacity|tank.*?hold|capacity).*?(?:is|=|equals?)\s*(\d+(?:\.\d+)?)\s*gallons?',
            r'(?:C|capacity)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*gallons?.*?(?:total|capacity|answer)',
            r'total.*?(\d+(?:\.\d+)?)\s*gallons?',
            r'(?:tank.*?holds?|capacity.*?is).*?(\d+(?:\.\d+)?)'
        ]
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, content.lower())
            if matches:
                return matches[-1].strip()
        final_answer_patterns = [
            r'final\s+answer[:\s]*(\d+(?:\.\d+)?)',
            r'answer[:\s]*(\d+(?:\.\d+)?)\s*(?:animals|total|$)',
            r'### final answer[:\s]*[^0-9]*(\d+(?:\.\d+)?)'
        ]
        for pattern in final_answer_patterns:
            match = re.search(pattern, content.lower())
            if match:
                return match.group(1).strip()
        if critic_approved:
            addition_patterns = [
                r'(\d+(?:\.\d+)?)\s*\+\s*\d+\s*\+.*?=\s*(\d+(?:\.\d+)?)',
                r'=\s*(\d+(?:\.\d+)?)\s*(?:animals|total|gallons?|$)',
                r'total\s*=.*?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:animals|total)',
            ]
            for pattern in addition_patterns:
                matches = re.findall(pattern, content.lower())
                if matches:
                    last_match = matches[-1]
                    if isinstance(last_match, tuple):
                        numbers = [float(x) for x in last_match if x]
                        if numbers:
                            max_number = max(numbers)
                            return str(int(max_number)) if max_number == int(max_number) else str(max_number)
                    else:
                        return last_match.strip()
        all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
        if all_numbers:
            return all_numbers[-1]
        return None
    def _execute_with_planner(self, agent_pool: dict, initial_budget: float, task_description: str, collaboration_pattern: dict = None, dataset_type: str = "gsm8k"):
        """
        UsePlannerAgentfor dynamic instruction scheduling（maintain existingexecute_hybridlogic）
        """
        print("\n" + "="*20 + " Hybrid Execution Started " + "="*20)
        current_agent_pool = copy.deepcopy(agent_pool['pool'])
        self.initial_pool = copy.deepcopy(agent_pool['pool'])
        execution_history = []
        total_cost = 0
        replans_due_to_critic = 0
        max_replans = 2
        planner_instance = self._get_agent_instance_from_pool('Planner', current_agent_pool)
        if not planner_instance:
            print("[!] Execution Failed: No Planner agent available in the pool.")
            return {"history": ["Execution failed: No Planner agent available in the pool."]}, 0
        self._consume_resource('Planner', planner_instance.config['config_level'], current_agent_pool)
        planner_resources = {}
        for role, levels in current_agent_pool.items():
            for level, data in levels.items():
                key = f"{role}_{level}"
                planner_resources[key] = {
                    "agent_choices": data.get('agents', []),
                    "slots_available": data.get('count', 0)
                }
        state_for_planner = {
            "task_description": task_description,
            "available_resources": planner_resources,
            "history": execution_history,
            "collaboration_pattern": collaboration_pattern
        }
        planner_prompt = planner_instance._create_prompt(state_for_planner)
        parsed_plan, metadata = planner_instance.execute(state_for_planner)
        planner_cost = metadata.get("actual_energy", 0)
        total_cost += planner_cost
        if collaboration_pattern and collaboration_pattern.get('name') == 'delegate_and_gather':
            decompressed_plan = []
            for step in parsed_plan:
                repeat_count = step.get("repeat", 1)
                if isinstance(repeat_count, int) and repeat_count > 1:
                    base_step = {k: v for k, v in step.items() if k != 'repeat'}
                    for _ in range(repeat_count):
                        decompressed_plan.append(copy.deepcopy(base_step))
                else:
                    decompressed_plan.append(step)
            parsed_plan = decompressed_plan
        if not parsed_plan:
            return self._force_final_answer(context={"task_description": task_description, "history": execution_history}, reason="Planner returned an empty or invalid plan.")
        plan = parsed_plan
        required_counts = self._count_agents_in_plan(plan)
        available_counts = self._count_available_agents_in_pool(current_agent_pool)
        plan_violates_constraints = False
        for agent_class, required_num in required_counts.items():
            available_num = available_counts.get(agent_class, 0)
            if required_num > available_num:
                plan_violates_constraints = True
                warning_msg = f"WARNING: Plan may violate resource constraints. Required {required_num} of '{agent_class}', but only {available_num} available. Will proceed with execution using fallback agents."
                execution_history.append({"step": 0, "agent": "System", "thought": warning_msg, "result": "WARNING: Resource constraint violation detected"})
        if plan_violates_constraints:
            print("[*] Proceeding with execution despite resource constraint warnings. The engine will use fallback agents when pool is exhausted.")
        print("\n--- Executing Plan ---")
        step_index = 0
        while step_index < len(plan):
            step = plan[step_index]
            print(f"\n--- Step {step_index + 1}/{len(plan)} ---")
            agent_specifier = step.get("agent")
            if not agent_specifier or agent_specifier == "Done":
                final_answer = step.get("input", {}).get("final_answer")
                if final_answer:
                    history_results = [h['result'] for h in execution_history]
                    final_answer_resolved = history_results[-1] if history_results else final_answer
                    execution_history.append({
                        "step": step_index + 1, "agent": "Done", "thought": "Plan finished.", "result": final_answer_resolved
                    })
                break
            agent_class_name, agent_role = self._get_class_and_role_from_specifier(agent_specifier, self.initial_pool)
            if not agent_role:
                error_msg = f"Could not determine role for agent specifier '{agent_specifier}'"
                execution_history.append({"step": step_index+1, "thought": error_msg, "result": "ERROR: Unknown agent specifier."})
                replan_result = self._attempt_replan_in_hybrid(
                    planner_instance, task_description, execution_history,
                    current_agent_pool, collaboration_pattern, error_msg,
                    plan, step_index, total_cost, dataset_type
                )
                if replan_result:
                    plan, total_cost = replan_result
                    continue
                else:
                    break
            agent_instance = self._get_agent_instance_from_pool(agent_class_name, current_agent_pool)
            if not agent_instance:
                agent_instance = self._get_agent_instance_from_pool(agent_role, current_agent_pool)
            if not agent_instance:
                agent_instance = self._create_fallback_agent_instance(agent_class_name)
                if not agent_instance:
                    error_msg = f"No agent available for class/role '{agent_class_name}/{agent_role}' and fallback creation failed"
                    execution_history.append({"step": step_index+1, "thought": error_msg, "result": "ERROR: Agent unavailable."})
                    replan_result = self._attempt_replan_in_hybrid(
                        planner_instance, task_description, execution_history,
                        current_agent_pool, collaboration_pattern, error_msg,
                        plan, step_index, total_cost, dataset_type
                    )
                    if replan_result:
                        plan, total_cost = replan_result
                        continue
                    else:
                        break
                else:
                    break
            else:
                self._consume_resource(agent_class_name, agent_instance.config['config_level'], current_agent_pool)
            try:
                raw_results = [h['result'] for h in execution_history if h.get('agent') != 'System']
                history_results = []
                for result in raw_results:
                    if isinstance(result, dict) and 'result' in result:
                        history_results.append(result['result'])
                    else:
                        history_results.append(result)
                state_for_executor = self._substitute_placeholders(step, history_results)
            except ValueError as e:
                print(f"[!] Execution Failed: Error substituting placeholders for step {step_index + 1}. Error: {e}")
                execution_history.append({"step": step_index+1, "thought": f"Error resolving placeholders: {e}", "result": "ERROR: Invalid dataflow reference."})
                break
            final_agent_state = state_for_executor.get('input', {})
            if agent_role == 'Critic':
                final_agent_state['task_description'] = task_description
                final_agent_state['history'] = copy.deepcopy(execution_history)
            final_agent_state['dataset_type'] = dataset_type
            agent_result, agent_metadata = agent_instance.execute(final_agent_state)
            agent_cost = agent_metadata.get("actual_energy", 0)
            total_cost += agent_cost
            if agent_role == 'Critic' and isinstance(agent_result, dict):
                result_to_store = agent_result
            else:
                result_to_store = agent_result
            step_summary = {
                "step": step_index + 1,
                "agent": agent_instance.id,
                "thought": step.get("thought", ""),
                "sub_task": step.get("sub_task", state_for_executor.get("input")),
                "result": result_to_store
            }
            execution_history.append(step_summary)
            if agent_role == 'Critic' and agent_result.get("critique") == 'INCORRECT':
                replans_due_to_critic += 1
                if replans_due_to_critic > max_replans:
                    final_context, final_cost = self._force_final_answer(
                        {"task_description": task_description, "history": execution_history},
                        "Maximum re-plan limit reached after repeated critic failures."
                    )
                    total_cost += final_cost
                    execution_history = final_context['history']
                    break
                remaining_budget = initial_budget - total_cost
                min_replan_budget = self._calculate_min_replan_budget()
                correction_task = agent_result.get('correction')
                if remaining_budget >= min_replan_budget:
                    new_agent_pool, is_feasible = self.low_level_instantiator.solve(
                        collaboration_pattern,
                        remaining_budget,
                        candidate_agent_ids=None
                    )
                    if is_feasible and new_agent_pool:
                        current_agent_pool = copy.deepcopy(new_agent_pool['pool'])
                        planner_instance_new = self._get_agent_instance_from_pool('Planner', current_agent_pool)
                        if not planner_instance_new:
                            print("[!] Re-provisioning failed to provide a Planner. Falling back to Level 2.")
                        else:
                            self._consume_resource('Planner', planner_instance_new.config['config_level'], current_agent_pool)
                            total_cost += new_agent_pool.get('cost', 0)
                            planner_resources = self._get_total_resources_for_planner(current_agent_pool)
                            replan_state = {
                                "task_description": task_description,
                                "available_resources": planner_resources,
                                "history": copy.deepcopy(execution_history),
                                "feedback_from_critic": agent_result.get('reason'),
                                "collaboration_pattern": collaboration_pattern
                            }
                            new_plan, replan_metadata = planner_instance_new.execute(replan_state)
                            total_cost += replan_metadata.get("actual_energy", 0)
                            if new_plan:
                                plan = plan[:step_index] + new_plan
                                continue 
                            else:
                                print("[!] Re-planning failed. Falling back to Level 3 (force output).")
                if correction_task and remaining_budget > 0:
                    corrector_instance = self._get_agent_instance_from_pool('Executor', current_agent_pool)
                    if corrector_instance:
                        self._consume_resource('Executor', corrector_instance.config['config_level'], current_agent_pool)
                        correction_input = {"instruction": correction_task}
                        correction_result, correction_metadata = corrector_instance.execute(correction_input)
                        correction_cost = correction_metadata.get("actual_energy", 0)
                        total_cost += correction_cost
                        execution_history.append({
                            "step": len(execution_history) + 1,
                            "agent": f"Corrector_{corrector_instance.config['config_level']}",
                            "thought": "Local correction based on critic feedback",
                            "result": correction_result,
                            "cost": correction_cost
                        })
                    step_index += 1
                    continue
                else:
                    print("[*] LEVEL 3: Forcing output due to insufficient resources or repeated failures.")
                final_context, final_cost = self._force_final_answer(
                    {"task_description": task_description, "history": execution_history},
                    "Forced output after critic feedback due to resource constraints."
                )
                total_cost += final_cost
                execution_history = final_context['history']
                break
            step_index += 1
        final_context = {"history": execution_history}
        print("\n" + "="*20 + " Hybrid Execution Finished " + "="*20)
        print(f"[*] Final Total Cost: {total_cost:.4f}")
        return final_context, total_cost
    def _attempt_replan_in_hybrid(self, planner_instance, task_description, execution_history, 
                                  current_agent_pool, collaboration_pattern, error_reason,
                                  plan, step_index, total_cost, dataset_type):
        """
        Attempts to re-plan when a step fails in execute_hybrid.
        Returns updated (plan, total_cost) tuple on success, or None on failure.
        """
        try:
            if not self._check_resource_availability("Planner", current_agent_pool):
                return None
            planner_resources = {}
            for role, levels in current_agent_pool.items():
                for level, data in levels.items():
                    key = f"{role}_{level}"
                    planner_resources[key] = {
                        "agent_choices": data.get('agents', []),
                        "slots_available": data.get('count', 0)
                    }
            replan_state = {
                "task_description": task_description,
                "available_resources": planner_resources,
                "history": copy.deepcopy(execution_history),
                "feedback_from_error": error_reason,
                "collaboration_pattern": collaboration_pattern
            }
            replan_planner = self._get_agent_instance_from_pool('Planner', current_agent_pool)
            if not replan_planner:
                print("[!] Failed to get planner instance for re-planning.")
                return None
            self._consume_resource('Planner', replan_planner.config['config_level'], current_agent_pool)
            new_plan, replan_metadata = replan_planner.execute(replan_state)
            replan_cost = replan_metadata.get("actual_energy", 0)
            total_cost += replan_cost
            if new_plan:
                updated_plan = plan[:step_index] + new_plan
                return updated_plan, total_cost
            else:
                print("[!] Re-planning failed, returned empty plan.")
                return None
        except Exception as e:
            print(f"[!] Re-planning failed with exception: {e}")
            return None
    def _get_class_and_role_from_specifier(self, specifier: str, pool: dict) -> Tuple[str | None, str | None]:
        """
        Robustly parses an agent specifier, which can be 'Role:ClassName' or just 'ClassName'.
        If only ClassName is provided, it searches the pool to find the corresponding role.
        """
        if ':' in specifier:
            parts = specifier.split(':', 1)
            return parts[1], parts[0]
        else:
            class_name = specifier
            for role, levels in pool.items():
                for level, data in levels.items():
                    if class_name in data.get('agents', []):
                        return class_name, role
        return specifier, None
    def _get_agent_instance_from_pool(self, agent_class_or_role: str, pool: dict, task_hint: str = None) -> BaseAgent | None:
        """
        Attempts to find an agent instance in the provided resource pool.
        It first tries to find by role (e.g., 'Planner', 'Executor'), then by class name.
        PRIORITY ORDER: High -> Low (always prefer High-level agents when available)
        For Executor role, intelligently selects between MathExecutor and CodeExecutor based on task content.
        """
        level_priority = ['High', 'Low']
        for role, levels in pool.items():
            if role == agent_class_or_role:
                for level in level_priority:
                    if level in levels and levels[level]['count'] > 0:
                        available_agents = levels[level]['agents']
                        if role == 'Executor' and len(available_agents) > 1 and task_hint:
                            agent_class_name = self._choose_best_executor(available_agents, task_hint)
                        else:
                            agent_class_name = available_agents[0]
                        return self._instantiate_agent_by_class_and_level(agent_class_name, level)
        for role, levels in pool.items():
            for level in level_priority:
                if level in levels and levels[level]['count'] > 0 and agent_class_or_role in levels[level]['agents']:
                    return self._instantiate_agent_by_class_and_level(agent_class_or_role, level)
        return None
    def _choose_best_executor(self, available_agents: list, task_hint: str) -> str:
        """
        Intelligently chooses the best executor agent based on task content.
        Args:
            available_agents: List of available agent class names (e.g., ['CodeExecutorAgent', 'MathExecutorAgent'])
            task_hint: Task description or instruction to help choose the right agent
        Returns:
            The class name of the best suited agent
        """
        if not task_hint:
            return available_agents[0]
        task_lower = task_hint.lower()
        has_math_executor = 'MathExecutorAgent' in available_agents
        has_code_executor = 'CodeExecutorAgent' in available_agents
        if has_math_executor and has_code_executor:
            math_indicators = [
                '+', '-', '*', '/', '=', 'plus', 'minus', 'times', 'divided',
                'calculate', 'compute', 'add', 'subtract', 'multiply', 'divide',
                'sum', 'total', 'cost', 'price', 'how much', 'how many',
                'solve this problem', 'what is', 'find the', 'determine'
            ]
            code_indicators = [
                'code', 'python', 'function', 'algorithm', 'loop', 'array',
                'list', 'dict', 'import', 'def ', 'for ', 'while ',
                'complex', 'advanced', 'computation'
            ]
            math_score = sum(1 for indicator in math_indicators if indicator in task_lower)
            code_score = sum(1 for indicator in code_indicators if indicator in task_lower)
            if any(simple in task_lower for simple in ['what is', 'calculate']) and any(op in task_lower for op in ['+', '-', '*', '/']):
                return 'MathExecutorAgent'
            if code_score > math_score:
                return 'CodeExecutorAgent'
            else:
                return 'MathExecutorAgent'
        if has_math_executor:
            return 'MathExecutorAgent'
        elif has_code_executor:
            return 'CodeExecutorAgent'
        return available_agents[0]
    def _consume_resource(self, agent_class_or_role: str, level: str, pool: dict):
        """
        Consumes a resource slot for a given agent class or role at a specific level.
        It first tries to find by role, then by agent class name.
        """
        for r, levels in pool.items():
            if r == agent_class_or_role and level in levels and levels[level]['count'] > 0:
                levels[level]['count'] -= 1
                return
        for role, levels in pool.items():
            if level in levels and agent_class_or_role in levels[level]['agents'] and levels[level]['count'] > 0:
                levels[level]['count'] -= 1
                return
    def _instantiate_agent_by_class_and_level(self, agent_class_name: str, level: str) -> BaseAgent | None:
        """
        Instantiates an agent instance based on its class name and level.
        It searches through the config_loader's agents to find the correct one.
        """
        for agent_id, agent_config in self.config_loader.agents.items():
            if agent_config['class_name'] == agent_class_name and agent_config['config_level'] == level:
                return self._create_agent_instance(agent_config)
        return None
    def _create_agent_instance(self, agent_config: Dict[str, Any]) -> BaseAgent:
        """
        Creates an agent instance from a configuration dictionary.
        """
        agent_id = agent_config['id']
        agent_instance = self.agent_factory.create_agent(agent_id, agent_config)
        return agent_instance
    def _prepare_step_prompt(self, step: dict, history: list) -> str:
        """
        [DEPRECATED] This method is a relic of an old design and is no longer used.
        The planner's output 'step' is now passed directly as the state to the
        executor's 'execute' method.
        """
        pass
    def execute_static(self, agent_pool: Dict[str, Any], initial_budget: float, task_description: str, collaboration_pattern: dict = None, dataset_type: str = "gsm8k") -> Tuple[Dict[str, Any], float]:
        """
        The `execute_static` method is now an alias for `execute_hybrid` as the new architecture
        is inherently hybrid (plan-then-execute).
        """
        print("Note: `execute_static` is now an alias for `execute_hybrid`.")
        return self.execute_hybrid(agent_pool, initial_budget, task_description, collaboration_pattern, dataset_type)
    def _extract_math_answer_from_result(self, result):
        """
        FromMATHdatasetExecutorextract answer from results，for aggregation strategy comparison
        Usemath_loaderstandard inMATHanswer extraction logic
        """
        try:
            from eapae_agent_sys.data_processing.math_loader import MATH_get_predict
            print('extract_math_answer_from_result', 'result:', result)
            if isinstance(result, dict):
                if 'numerical_answer' in result and result['numerical_answer'] != 'ERROR_PARSING':
                    content = result['numerical_answer']
                elif 'result' in result:
                    content = result['result']
                    content = '\\boxed{' + content + '}'
                    extracted = MATH_get_predict(content)
                    return extracted.strip() if extracted else content.strip()
                else:
                    content = str(result)
            else:
                content = str(result)
            return content.strip() if content else ""
        except ImportError:
            import re
            if isinstance(result, dict) and 'result' in result:
                content = result['result']
            else:
                content = str(result)
            numbers = re.findall(r'\d+(?:\.\d+)?', content)
            return numbers[-1].strip() if numbers else content.strip()