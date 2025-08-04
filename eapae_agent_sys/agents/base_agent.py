from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from eapae_agent_sys.utils.llm_api import llm_api
from eapae_agent_sys.utils.config_loader import config_loader
class BaseAgent(ABC):
    """
    AllAgentabstract base class。
    It definesAgentunified interface and basic properties。
    """
    def __init__(self, agent_config_id: str, config: Dict = None):
        """
        @param agent_config_id: Agentunique identifier for configuration, e.g., "PlannerAgent_GPT4o_High".
                                ThisIDused to lookup detailed information from configuration。
        @param config: (Optional) a configuration dictionary。if provided，will directly use this configuration，
                       mainly used forprofilingscript，to avoid circular dependencies。
        """
        self.id = agent_config_id
        if config:
            self.config = config
        else:
            try:
                self.config = self._get_config_from_loader(agent_config_id)
            except ValueError:
                print(f"Warning: Config for '{agent_config_id}' not found. Using empty config. This is expected during profiling.")
                self.config = {}
        self.agent_class_name = self.config.get('class_name', self.__class__.__name__)
        self.llm_provider = self.config.get('provider')
        self.model_name = self.config.get('model_name')
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.energy_model_coeffs = {
            'A': self.config.get('cost_coeff_A', 0), 
            'B': self.config.get('cost_coeff_B', 0)
        }
        self.config_level = self.config.get('config_level', 'default').lower()
        self.llm_api = llm_api
    def _get_config_from_loader(self, agent_config_id: str) -> Dict:
        """Inagent_library.ymllookup and return specificAgentconfiguration。"""
        if agent_config_id is None:
            return {}
        all_agents = config_loader.agents.get('agents', [])
        for agent_cfg in all_agents:
            if agent_cfg['id'] == agent_config_id:
                return agent_cfg
        raise ValueError(f"Agent config ID '{agent_config_id}' not found in agent library.")
    @abstractmethod
    def execute(self, state: Dict) -> Tuple[Any, Dict]:
        """
        executeAgentcore logic。This is an abstract method，must be implemented by subclasses。
        @param state: dictionary containing task context information。
                      for example: {'task_description': "...", 'history': [...]}
        @return: a tuple，containing:
                 - agentoutput，usuallystr，but can also be other types（such asdict）。
                 - dictionary containing metadata (dict), must contain 'usage' and 'actual_energy'。
                   for example: {'usage': {'prompt_tokens': 100, 'completion_tokens': 50}, 'actual_energy': 123.45}
        """
        pass
    def _calculate_actual_energy(self, usage: Dict) -> float:
        """Based ontokenusage and energy model coefficients calculate actual energy consumption。"""
        if not usage or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            return 0.0
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        actual_energy = (self.energy_model_coeffs.get('A', 0) * prompt_tokens + 
                         self.energy_model_coeffs.get('B', 0) * completion_tokens)
        return actual_energy 