from .base_agent import BaseAgent
from typing import Dict, Tuple
class PoetryAgent(BaseAgent):
    """
    Generate a short poem or haiku based on given topic。
    """
    def execute(self, state: Dict) -> Tuple[str, Dict]:
        theme = state.get("theme", "the wind")
        prompt = f"Write a short, three-line poem (haiku) about {theme}."
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        return response, {"usage": usage, "actual_energy": actual_energy}
class MarketingCopyAgent(BaseAgent):
    """
    Write compelling marketing slogan for product or service。
    """
    def execute(self, state: Dict) -> Tuple[str, Dict]:
        product = state.get("product", "a new coffee machine")
        prompt = f"Write a catchy, one-sentence marketing slogan for {product}."
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        return response, {"usage": usage, "actual_energy": actual_energy}
class HistoricalFactAgent(BaseAgent):
    """
    Provide random or queried historical fact for given date or topic。
    """
    def execute(self, state: Dict) -> Tuple[str, Dict]:
        topic = state.get("topic", "the moon landing")
        prompt = f"Provide a single, interesting historical fact about {topic}."
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        return response, {"usage": usage, "actual_energy": actual_energy}
class CodeRefactorAgent(BaseAgent):
    """
    Refactor givenPythoncode block to improve style and efficiency，without changing its logic。
    """
    def execute(self, state: Dict) -> Tuple[str, Dict]:
        code = state.get("code", "def f(x): return x*2")
        prompt = (
            "You are a senior Python developer. Refactor the following code to improve its style, readability, "
            "and efficiency, without changing its underlying logic or output. Provide only the refactored code.\n\n"
            f"Original Code:\n```python\n{code}\n```"
        )
        response, usage = self.llm_api.call(
            provider=self.llm_provider,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        actual_energy = self._calculate_actual_energy(usage)
        return response, {"usage": usage, "actual_energy": actual_energy} 