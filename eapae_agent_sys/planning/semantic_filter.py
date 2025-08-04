from ..utils.config_loader import ConfigLoader
class SemanticFilter:
    """
    Stage 1: Semantic Candidate Filtering.
    Uses predefined rules to filter relevant agents for a given task description.
    """
    def __init__(self, config_loader: ConfigLoader, llm_api=None):
        """
        Initializes the SemanticFilter.
        Args:
            config_loader: An instance of ConfigLoader holding all configurations.
            llm_api: Not used anymore, kept for backward compatibility.
        """
        self.agent_library = config_loader.agents
    def filter_candidates(self, task_description: str, collaboration_pattern: dict = None) -> list[str]:
        """
        Filters the global agent library to a smaller candidate set using predefined rules.
        It can also receive the collaboration pattern to make sure essential agents for that
        pattern are included. Updated to support topology-driven architecture.
        Args:
            task_description: The natural language description of the task (used for dataset detection).
            collaboration_pattern: The selected collaboration pattern for the task.
        Returns:
            A list of agent IDs deemed relevant.
        """
        agent_library_keys = list(self.agent_library.keys())
        is_math_dataset = any('MathExecutorAgent' in key and 'MBPPCodeSolveAgent' not in str(agent_library_keys) for key in agent_library_keys)
        is_mbpp_dataset = any('MBPPCodeSolveAgent' in key for key in agent_library_keys)
        if is_mbpp_dataset:
            essential_base_ids = set()
            if collaboration_pattern:
                topology_type = collaboration_pattern.get('topology_type', 'dynamic') 
                pattern_name = collaboration_pattern.get('name', '')
                if topology_type == 'dynamic':
                    essential_base_ids.add("PlannerAgent")
                if 'feedback' in pattern_name:
                    essential_base_ids.add("CodeCriticAgent")
                essential_base_ids.add("MBPPCodeSolveAgent")
            else:
                essential_base_ids.update({
                    "PlannerAgent",
                    "MBPPCodeSolveAgent",
                    "CodeCriticAgent"
                })
            full_candidate_ids = []
            for agent_config in self.agent_library.values():
                base_agent_name = agent_config.get('class_name')
                if base_agent_name in essential_base_ids:
                    full_candidate_ids.append(agent_config.get('id'))
            return full_candidate_ids
        elif is_math_dataset:
            essential_base_ids = set()
            if collaboration_pattern:
                topology_type = collaboration_pattern.get('topology_type', 'dynamic')
                pattern_name = collaboration_pattern.get('name', '')
                if topology_type == 'dynamic':
                    essential_base_ids.add("PlannerAgent")
                if 'feedback' in pattern_name or 'critic' in pattern_name.lower():
                    essential_base_ids.add("SolutionCriticAgent")
                essential_base_ids.add("MathExecutorAgent")
            else:
                essential_base_ids.update({
                    "PlannerAgent", 
                    "MathExecutorAgent",
                    "SolutionCriticAgent"
                })
            full_candidate_ids = []
            for agent_config in self.agent_library.values():
                base_agent_name = agent_config.get('class_name')
                if base_agent_name in essential_base_ids:
                    full_candidate_ids.append(agent_config['id'])
            return list(set(full_candidate_ids))
        # For all other datasets (GSM8K, MATH, etc.), use the same predefined logic as MATH
        essential_base_ids = set()
        if collaboration_pattern:
            topology_type = collaboration_pattern.get('topology_type', 'dynamic')
            pattern_name = collaboration_pattern.get('name', '')
            if topology_type == 'dynamic':
                essential_base_ids.add("PlannerAgent")
            if 'feedback' in pattern_name or 'critic' in pattern_name.lower():
                essential_base_ids.add("SolutionCriticAgent")
            essential_base_ids.add("MathExecutorAgent")
        else:
            essential_base_ids.update({
                "PlannerAgent", 
                "MathExecutorAgent",
                "SolutionCriticAgent"
            })
        
        full_candidate_ids = []
        for agent_config in self.agent_library.values():
            base_agent_name = agent_config.get('class_name')
            if base_agent_name in essential_base_ids:
                full_candidate_ids.append(agent_config['id'])
        return list(set(full_candidate_ids)) 