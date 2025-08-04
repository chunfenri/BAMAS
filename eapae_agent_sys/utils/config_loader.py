import yaml
import json
import os
from typing import Dict, Any
class ConfigLoader:
    _instances = {}
    def __new__(cls, config_dir="configs", agent_library_file=None, training_params_file=None):
        if agent_library_file is None:
            agent_library_file = "0_agent_library.yml"
        if training_params_file is None:
            training_params_file = "3_training_params.yml"
        instance_key = f"{config_dir}_{agent_library_file}_{training_params_file}"
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(ConfigLoader, cls).__new__(cls)
            cls._instances[instance_key].config_dir = config_dir
            cls._instances[instance_key].agent_library_file = agent_library_file
            cls._instances[instance_key].training_params_file = training_params_file
            cls._instances[instance_key].load_all()
        return cls._instances[instance_key]
    def load_all(self):
        """
        Loads all configuration files at once and processes them into their final,
        usable format.
        """
        raw_agents = self._load_yaml(self.agent_library_file)
        self.agents = {agent['id']: agent for agent in raw_agents.get('agents', [])}
        self.roles = self._load_yaml("1_role_compatibility.yml")
        self.patterns = self._load_json("2_collaboration_patterns.json")
        self.params = self._load_yaml(self.training_params_file)
        self.secrets = self._load_yaml("secrets.yml", ignore_not_found=True)
        if not self.secrets:
            self.secrets = {'api_keys': {}}
    def _load_yaml(self, filename, ignore_not_found=False):
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            if ignore_not_found:
                return None
            print(f"Error: Required config file not found at {filepath}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filepath}: {e}")
            raise
    def _load_json(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Required config file not found at {filepath}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {filepath}: {e}")
            raise
config_loader = ConfigLoader() 