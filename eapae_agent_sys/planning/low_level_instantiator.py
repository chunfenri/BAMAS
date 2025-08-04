import pulp
import datetime
import re
from ..utils.config_loader import ConfigLoader
MAX_EXPANDABLE_INSTANCES = 5 
class LowLevelInstantiator:
    """
    Stage 2.2: Low-Level Dynamic Role Instantiator.
    Solves an ILP problem to instantiate a specific execution graph from a
    collaboration pattern, a set of candidate agents, and an energy budget.
    """
    def __init__(self, config_loader: ConfigLoader):
        """
        Initializes the LowLevelInstantiator.
        Args:
            config_loader: An instance of ConfigLoader holding all configurations.
        """
        self.agents_config = config_loader.agents
        self.ilp_avg_prompt_tokens = config_loader.params.get('ilp_solver', {}).get('avg_prompt_tokens', 250)
        self.agent_to_roles = {}
        self.role_to_agents = {}
        for agent_id, agent_conf in self.agents_config.items():
            agent_class = agent_conf.get('class_name')
            role = agent_conf.get('role')
            if not agent_class or not role:
                continue
            if role not in self.role_to_agents:
                self.role_to_agents[role] = []
            if agent_class not in self.role_to_agents[role]:
                self.role_to_agents[role].append(agent_class)
            if agent_class not in self.agent_to_roles:
                self.agent_to_roles[agent_class] = []
            if role not in self.agent_to_roles[agent_class]:
                self.agent_to_roles[agent_class].append(role)
        self.patterns_config = {p['name']: p for p in config_loader.patterns['patterns']}
        self.performance_tiers = {
            'High': 1,
            'Low': 2
        }
    def _calculate_lexicographic_weights(self, budget: float, groups: dict) -> dict:
        """
        Calculate lexicographic weights using the recursive formula from the paper:
        W_i = 1 + ∑(j=i+1 to L) (W_j · ⌊B/c_j⌋)
        This ensures that selecting a higher-tier LLM always outweighs 
        any budget-feasible combination of lower-tier models.
        """
        tier_costs = {}
        for (role, level), agents in groups.items():
            tier = self.performance_tiers.get(level, 999)
            if tier not in tier_costs:
                total_cost = 0
                for agent in agents:
                    max_completion_tokens = agent.get('max_tokens', 256)
                    cost = (
                        agent.get('cost_coeff_A', 0) * self.ilp_avg_prompt_tokens +
                        agent.get('cost_coeff_B', 0) * max_completion_tokens
                    )
                    total_cost += cost
                tier_costs[tier] = total_cost / len(agents)
        sorted_tiers = sorted(tier_costs.keys())
        weights = {}
        if sorted_tiers:
            weights[sorted_tiers[-1]] = 1.0
            for i in range(len(sorted_tiers) - 2, -1, -1):
                tier_i = sorted_tiers[i]
                weight_sum = 0.0
                for j in range(i + 1, len(sorted_tiers)):
                    tier_j = sorted_tiers[j]
                    c_j = tier_costs[tier_j]
                    if c_j > 0:
                        affordable_count = int(budget / c_j)
                        weight_sum += weights[tier_j] * affordable_count
                weights[tier_i] = 1.0 + weight_sum
        return weights
    def get_compatible_agents_for_pattern(self, pattern: dict) -> list:
        """
        Finds all agent configurations from the library that can play any role
        defined in the given collaboration pattern.
        """
        if not pattern:
            return []
        roles_in_pattern = {role['name'] for role in pattern.get('roles', [])}
        required_agent_classes = set()
        for role_name in roles_in_pattern:
            required_agent_classes.update(self.role_to_agents.get(role_name, []))
        compatible_agents = [
            agent_conf for agent_conf in self.agents_config.values()
            if agent_conf['class_name'] in required_agent_classes
        ]
        return compatible_agents
    def solve(self, pattern_or_name, budget: float, candidate_agent_ids: list[str] | None = None) -> tuple[dict | None, bool]:
        """
        Builds and solves the ILP problem to find an optimal and feasible agent allocation.
        This new version allocates "slots" for role/level combinations and then fills
        them with the best available agents. Now supports topology-specific optimization.
        Args:
            pattern_or_name: The collaboration pattern to use.
            budget: The maximum allowable cost for the team.
            candidate_agent_ids: A list of agent IDs to consider. If None, all agents are considered.
        """
        timestamp = datetime.datetime.now()
        if isinstance(pattern_or_name, str):
            pattern = self.patterns_config.get(pattern_or_name)
        elif isinstance(pattern_or_name, dict):
            pattern = pattern_or_name
        else:
            return None, False
        if not pattern:
            return None, False
        topology_type = pattern.get('topology_type', 'dynamic')
        if topology_type == 'fixed':
            return self._solve_fixed_topology(pattern, budget, candidate_agent_ids)
        else:
            return self._solve_dynamic_topology(pattern, budget, candidate_agent_ids)
    def _solve_fixed_topology(self, pattern: dict, budget: float, candidate_agent_ids: list[str] | None = None):
        """
        For fixed topologyILPsolving：focus on optimizing variable role counts and configuration levels
        Fixed topology does not requirePlanner，focus on optimizingExecutorrole counts and configuration allocation
        """
        timestamp = datetime.datetime.now()
        prob = pulp.LpProblem("Fixed_Topology_Agent_Allocation", pulp.LpMaximize)
        groups = {}
        EXCLUDED_ROLES_FOR_FIXED = {'Planner'}
        if pattern.get('name') == 'feedback_topology':
            MANDATORY_ROLES_FOR_FIXED = {'Executor'}
        else:
            MANDATORY_ROLES_FOR_FIXED = {'Executor', 'OutputFormatter'}
        candidate_set = set(candidate_agent_ids) if candidate_agent_ids is not None else None
        for agent_conf in self.agents_config.values():
            agent_class_name = agent_conf['class_name']
            config_level = agent_conf['config_level']
            compatible_roles = self.agent_to_roles.get(agent_class_name, [])
            for role_name in compatible_roles:
                if role_name in EXCLUDED_ROLES_FOR_FIXED:
                    continue
                if role_name not in {r['name'] for r in pattern['roles']}:
                    continue
                is_mandatory = role_name in MANDATORY_ROLES_FOR_FIXED
                is_in_candidates = candidate_set is not None and agent_conf['id'] in candidate_set
                if candidate_set is not None:
                    should_include = is_in_candidates
                else:
                    should_include = is_mandatory
                if should_include:
                    group_key = (role_name, config_level)
                    if group_key not in groups:
                        groups[group_key] = []
                    if agent_conf not in groups[group_key]:
                        groups[group_key].append(agent_conf)
        if not groups:
            return None, False
        y = pulp.LpVariable.dicts("y",
                                  [(g_key[0], g_key[1], k) for g_key in groups for k in range(MAX_EXPANDABLE_INSTANCES)],
                                  cat='Binary')
        lexicographic_weights = self._calculate_lexicographic_weights(budget, groups)
        def get_fixed_topology_priority_weight(role, level):
            """Returns lexicographic priority weight for fixed topology optimization."""
            tier = self.performance_tiers.get(level, 999)
            base_weight = lexicographic_weights.get(tier, 1.0)
            if role == 'Executor':
                if level == 'High':
                    return base_weight * 1.01
                else:
                    return base_weight * 1.00
            elif role == 'OutputFormatter':
                return base_weight * 0.95
            elif role == 'Critic':
                return base_weight * 1.005
            else:
                return base_weight * 0.90
        objective_terms = []
        for g_key, agents in groups.items():
            for k in range(MAX_EXPANDABLE_INSTANCES):
                avg_performance = sum(agent.get('performance_p', 0) for agent in agents) / len(agents)
                priority_weight = get_fixed_topology_priority_weight(g_key[0], g_key[1])
                coefficient = avg_performance * priority_weight
                objective_terms.append(f"y[{g_key},{k}] * {coefficient}")
        objective = pulp.lpSum(
            y[(g_key[0], g_key[1], k)] * (
                (sum(agent.get('performance_p', 0) for agent in agents) / len(agents)) * 
                get_fixed_topology_priority_weight(g_key[0], g_key[1])
            )
            for g_key, agents in groups.items() for k in range(MAX_EXPANDABLE_INSTANCES)
        )
        prob += objective
        def calculate_avg_estimated_cost(agents_in_group):
            if not agents_in_group:
                return 0
            total_cost = 0
            for agent_config in agents_in_group:
                max_completion_tokens = agent_config.get('max_tokens', 256)
                total_cost += (
                    agent_config.get('cost_coeff_A', 0) * self.ilp_avg_prompt_tokens +
                    agent_config.get('cost_coeff_B', 0) * max_completion_tokens
                )
            return total_cost / len(agents_in_group)
        energy_constraint = pulp.lpSum(
            y[(g_key[0], g_key[1], k)] * calculate_avg_estimated_cost(agents)
            for g_key, agents in groups.items()
            for k in range(MAX_EXPANDABLE_INSTANCES)
        )
        prob += energy_constraint <= budget
        for role_info in pattern['roles']:
            role_name = role_info['name']
            if role_name in EXCLUDED_ROLES_FOR_FIXED:
                continue
            matching_groups = [(r, level) for (r, level), agents in groups.items() if r == role_name]
            all_instances_for_role = pulp.lpSum(
                y[(role_name, level, k)]
                for (r, level), agents in groups.items() if r == role_name
                for k in range(MAX_EXPANDABLE_INSTANCES)
            )
            if not all_instances_for_role: 
                continue
            min_count = role_info.get('min_count', 1 if role_name in MANDATORY_ROLES_FOR_FIXED else 0)
            max_count = role_info.get('max_count', MAX_EXPANDABLE_INSTANCES)
            if role_info['type'] == 'Singleton':
                max_count = min(max_count, 1)
            if min_count > 0:
                constraint_name = f"MinConstraint_{role_name}"
                constraint = all_instances_for_role >= min_count
                prob += (constraint, constraint_name)
            max_constraint_name = f"MaxConstraint_{role_name}"
            max_constraint = all_instances_for_role <= max_count
            prob += (max_constraint, max_constraint_name)
        if pattern.get('name') == 'feedback_topology':
            high_critic_count = pulp.lpSum(
                y[('Critic', 'High', k)]
                for (r, level), agents in groups.items() if r == 'Critic' and level == 'High'
                for k in range(MAX_EXPANDABLE_INSTANCES)
            )
            high_executor_count = pulp.lpSum(
                y[('Executor', 'High', k)]
                for (r, level), agents in groups.items() if r == 'Executor' and level == 'High'
                for k in range(MAX_EXPANDABLE_INSTANCES)
            )
            high_critic_constraint = high_critic_count <= high_executor_count
            prob += (high_critic_constraint, "HighCriticRequiresHighExecutor")
            all_critic_count = pulp.lpSum(
                y[('Critic', level, k)]
                for (r, level), agents in groups.items() if r == 'Critic'
                for k in range(MAX_EXPANDABLE_INSTANCES)
            )
            all_executor_count = pulp.lpSum(
                y[('Executor', level, k)]
                for (r, level), agents in groups.items() if r == 'Executor'
                for k in range(MAX_EXPANDABLE_INSTANCES)
            )
            all_critic_constraint = all_critic_count <= all_executor_count
            prob += (all_critic_constraint, "CriticCountLEExecutorCount")
            print(f"  [{timestamp}] [ILP Solver] Added feedback constraint: Total Critic count <= Total Executor count")
        return self._solve_and_process_result(prob, groups, budget, timestamp, "Fixed Topology", pattern)
    def _solve_dynamic_topology(self, pattern: dict, budget: float, candidate_agent_ids: list[str] | None = None):
        """
        For dynamic topologyILPsolving：maintain existing logic
        Dynamic topology requiresPlanner，use original constraint logic
        """
        timestamp = datetime.datetime.now()
        prob = pulp.LpProblem("Agent_Allocation", pulp.LpMaximize)
        groups = {}
        MANDATORY_ROLES = {'Planner', 'Executor'}
        if pattern['name'] in ['reflection_loop']:
            MANDATORY_ROLES.add('Critic')
        candidate_set = set(candidate_agent_ids) if candidate_agent_ids is not None else None
        for agent_conf in self.agents_config.values():
            agent_class_name = agent_conf['class_name']
            config_level = agent_conf['config_level']
            compatible_roles = self.agent_to_roles.get(agent_class_name, [])
            for role_name in compatible_roles:
                if role_name not in {r['name'] for r in pattern['roles']}:
                    continue
                is_mandatory = role_name in MANDATORY_ROLES
                is_in_candidates = candidate_set is not None and agent_conf['id'] in candidate_set
                if candidate_set is not None:
                    should_include = is_in_candidates
                else:
                    should_include = is_mandatory
                if should_include:
                    group_key = (role_name, config_level)
                    if group_key not in groups:
                        groups[group_key] = []
                    if agent_conf not in groups[group_key]:
                        groups[group_key].append(agent_conf)
        if not groups:
            return None, False
        y = pulp.LpVariable.dicts("y",
                                  [(g_key[0], g_key[1], k) for g_key in groups for k in range(MAX_EXPANDABLE_INSTANCES)],
                                  cat='Binary')
        lexicographic_weights = self._calculate_lexicographic_weights(budget, groups)
        def get_priority_weight(role, level):
            """Returns lexicographic priority weight based on agent role and level."""
            tier = self.performance_tiers.get(level, 999)
            base_weight = lexicographic_weights.get(tier, 1.0)
            if role == 'Planner':
                return base_weight * 1.001
            else:
                return base_weight
        objective = pulp.lpSum(
            y[(g_key[0], g_key[1], k)] * (
                (sum(agent['performance_p'] for agent in agents) / len(agents)) * 
                get_priority_weight(g_key[0], g_key[1])
            )
            for g_key, agents in groups.items() for k in range(MAX_EXPANDABLE_INSTANCES)
        )
        prob += objective
        def calculate_avg_estimated_cost(agents_in_group):
            if not agents_in_group:
                return 0
            total_cost = 0
            for agent_config in agents_in_group:
                max_completion_tokens = agent_config.get('max_tokens', 256)
                total_cost += (
                    agent_config.get('cost_coeff_A', 0) * self.ilp_avg_prompt_tokens +
                    agent_config.get('cost_coeff_B', 0) * max_completion_tokens
                )
            return total_cost / len(agents_in_group)
        energy_constraint = pulp.lpSum(
            y[(g_key[0], g_key[1], k)] * calculate_avg_estimated_cost(agents)
            for g_key, agents in groups.items()
            for k in range(MAX_EXPANDABLE_INSTANCES)
        )
        prob += energy_constraint <= budget
        for role_info in pattern['roles']:
            role_name = role_info['name']
            all_instances_for_role = pulp.lpSum(
                y[(role_name, level, k)]
                for (r, level), agents in groups.items() if r == role_name
                for k in range(MAX_EXPANDABLE_INSTANCES)
            )
            if not all_instances_for_role: 
                continue
            if role_name in MANDATORY_ROLES:
                 prob += (all_instances_for_role >= 1, f"Constraint_Mandatory_{role_name}")
            if role_info['type'] == 'Singleton':
                prob += (all_instances_for_role <= 1)
        return self._solve_and_process_result(prob, groups, budget, timestamp, "Dynamic Topology", pattern)
    def _solve_and_process_result(self, prob, groups, budget, timestamp, topology_name, pattern=None):
        """
        Shared solving and result processing logic
        """
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if prob.status == pulp.LpStatusOptimal:
            resource_pool = {}
            solution_cost = 0
            selected_nodes_log = []
            y = prob.variablesDict()
            for var_name, var in sorted(y.items())[:10]:
                print(f"    {var_name} = {var.varValue}")
            for var_name, var in y.items():
                if var.varValue is not None and var.varValue > 0:
                    if var_name.startswith("y_(") and var_name.endswith(")"):
                        content = var_name[3:-1]
                        parts = content.split(',_')
                        if len(parts) == 3:
                            role = parts[0].strip("'")
                            level = parts[1].strip("'")
                            k = int(parts[2])
                            group_key = (role, level)
                            agents_in_group = groups.get(group_key, [])
                            if not agents_in_group: 
                                continue
                            def calculate_avg_estimated_cost(agents_in_group):
                                if not agents_in_group:
                                    return 0
                                total_cost = 0
                                for agent_config in agents_in_group:
                                    max_completion_tokens = agent_config.get('max_tokens', 256)
                                    total_cost += (
                                        agent_config.get('cost_coeff_A', 0) * self.ilp_avg_prompt_tokens +
                                        agent_config.get('cost_coeff_B', 0) * max_completion_tokens
                                    )
                                return total_cost / len(agents_in_group)
                            cost = calculate_avg_estimated_cost(agents_in_group)
                            solution_cost += cost
                            if role not in resource_pool: 
                                resource_pool[role] = {}
                            if level not in resource_pool[role]:
                                all_choices = [ac['class_name'] for ac in agents_in_group]
                                resource_pool[role][level] = {
                                    'count': 0, 
                                    'agents': sorted(list(set(all_choices))), 
                                    'cost_per_instance': cost
                                }
                            resource_pool[role][level]['count'] += 1
                            agent_choices_str = ", ".join(resource_pool[role][level]['agents'])
                            selected_nodes_log.append(
                                f"    - Slot: {role}_{level}_{k} | Choices: {agent_choices_str:<20} | Avg Cost: {cost:>6.2f}"
                            )
            print(f"  [{timestamp}] [ILP Solver] {topology_name} optimal team found with total cost {solution_cost:.2f} / {budget}:")
            if selected_nodes_log:
                for line in sorted(selected_nodes_log):
                    print(line)
            else:
                print("    (No agents selected)")
            topology_type = "fixed" if "Fixed" in topology_name else "dynamic"
            if topology_type == "fixed":
                required_roles = {'Executor'}
                if pattern and pattern.get('name') == 'feedback_topology':
                    required_roles.add('Critic')
            else:
                required_roles = {'Planner', 'Executor'}
            if not required_roles.issubset(resource_pool.keys()):
                return None, False
            return {'pool': resource_pool, 'status': 'Optimal', 'cost': solution_cost}, True
        else:
            print(f"  [{timestamp}] [ILP Solver] {topology_name} optimization failed")
            return None, False