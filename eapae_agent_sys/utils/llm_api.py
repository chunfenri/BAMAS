import os
import openai
import requests
import json
import datetime
import yaml
from typing import Tuple, Dict
from .config_loader import config_loader
class LLM_API:
    _secrets = None
    @staticmethod
    def _load_secrets():
        if LLM_API._secrets is None:
            try:
                with open("configs/secrets.yml", 'r') as f:
                    LLM_API._secrets = yaml.safe_load(f)
            except (FileNotFoundError, yaml.YAMLError):
                LLM_API._secrets = {}
        return LLM_API._secrets
    @staticmethod
    def _get_api_key(provider: str) -> str:
        """Tries to get API key from secrets file first, then environment variables."""
        secrets = LLM_API._load_secrets()
        if secrets:
            key = secrets.get('api_keys', {}).get(provider)
            if key is not None:
                if key == "EMPTY" and provider.startswith('local'):
                    return "dummy-key-for-local-service"
                elif key and f"YOUR_{provider.upper()}_API_KEY_HERE" not in key:
                    return key
        env_var_name = f"{provider.upper()}_API_KEY"
        key = os.environ.get(env_var_name)
        if key:
            return key
        if provider.startswith('local'):
            return "dummy-key-for-local-service"
        raise ValueError(
            f"API key for '{provider}' not found. "
            f"Please add it to configs/secrets.yml or set the {env_var_name} environment variable."
        )
    @staticmethod
    def _get_api_base(provider: str) -> str:
        """Tries to get API base URL from secrets file, then environment variables."""
        secrets = LLM_API._load_secrets()
        if secrets:
            base_url = secrets.get('api_bases', {}).get(provider)
            if base_url:
                return base_url
        env_var_name = f"{provider.upper()}_API_BASE"
        base_url = os.environ.get(env_var_name)
        if base_url:
            return base_url
        return None
    def __init__(self):
        self.api_keys = config_loader.secrets.get('api_keys', {})
        self.local_tokenizers = {}
    @staticmethod
    def call(provider: str, model: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
        """
        Unified API call entry point.
        """
        print("===================== LLM PROMPT START =====================")
        print(prompt)
        print("====================== LLM PROMPT END ======================")
        api_start_time = datetime.datetime.now()
        try:
            if provider == 'openai':
                response = LLM_API._call_openai(model=model, prompt=prompt, max_tokens=max_tokens)
                text_response = response['choices'][0]['message']['content']
                usage = response['usage']
            elif provider == 'deepseek':
                response = LLM_API._call_deepseek(model=model, prompt=prompt, max_tokens=max_tokens)
                text_response = response['choices'][0]['message']['content']
                usage = response['usage']
            elif provider == 'local_custom':
                response = LLM_API._call_local_custom(model=model, prompt=prompt, max_tokens=max_tokens)
                text_response = response['choices'][0]['message']['content']
                usage = response['usage']
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            api_end_time = datetime.datetime.now()
            print(f"    [{api_end_time}] [LLM_API] Call finished in {(api_end_time - api_start_time).total_seconds():.2f}s.")
            print("==================== LLM RESPONSE START ====================")
            print(text_response)
            print("===================== LLM RESPONSE END =====================")
            return text_response, usage
        except Exception as e:
            print(f"    [{datetime.datetime.now()}] [LLM_API] API call failed: {e}")
            if provider.startswith('local'):
                if "parse" in prompt.lower() or "extract" in prompt.lower():
                    fallback_response = "Unable to parse - local service unavailable"
                elif "format" in prompt.lower():
                    fallback_response = "#### Unable to format - local service unavailable"
                else:
                    fallback_response = "Unable to process - local service unavailable"
                return fallback_response, {"prompt_tokens": 0, "completion_tokens": 0}
            else:
                return f"Error: API call failed. Details: {e}", {"prompt_tokens": 0, "completion_tokens": 0}
    @staticmethod
    def _call_openai(model: str, prompt: str, max_tokens: int) -> Dict:
        try:
            api_key = LLM_API._get_api_key('openai')
            api_base = LLM_API._get_api_base('openai')
            client = openai.OpenAI(api_key=api_key, base_url=api_base)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return {
                'choices': [{'message': {'content': response.choices[0].message.content.strip()}}],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")
    @staticmethod
    def _call_deepseek(model, prompt, max_tokens):
        api_key = LLM_API._get_api_key("deepseek")
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    def _call_local_qwen(self, model: str, prompt: str, max_tokens: int) -> Tuple[str, Dict]:
        try:
            api_key = self._get_api_key('local_qwen')
            client = openai.OpenAI(api_key=api_key, base_url="http://127.0.0.1:8000/v1")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            content = response.choices[0].message.content
            return content.strip(), usage
        except Exception as e:
            raise RuntimeError(f"Local Qwen API call failed: {e}")
    @staticmethod
    def _call_local_custom(model, prompt, max_tokens):
        url = "http://127.0.0.1:8000/generate"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        generated_text = result.get("generated_text", "")
        prompt_tokens = len(prompt.split()) * 1.3
        completion_tokens = len(generated_text.split()) * 1.3
        return {
            "choices": [{"message": {"content": generated_text}}],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens)
            }
        }
llm_api = LLM_API()
if __name__ == "__main__":
    print("so that 'configs/secrets.yml' can be correctly located.")
    try:
        response_text, usage_data = LLM_API.call(
            provider='openai',
            model='gpt-4.1-nano',
            prompt='This is a test request. Please reply with the single word: "Success".',
            max_tokens=20
        )
        print("\n" + "="*20 + " TEST RESULT " + "="*20)
        print(f"Full Response: '{response_text}'")
        print(f"Token Usage: {usage_data}")
        print("="*53)
        if "success" in response_text.lower() and usage_data.get('completion_tokens', 0) > 0:
            print("\n Verification Successful: The API call via the proxy was successful.")
        else:
            print("\n Verification Warning: The API call returned, but the response was not the expected 'Success'.")
            print("   Please check the proxy server's logs and your API key configuration.")
    except Exception as e:
        print(f"\n Verification Failed: An error occurred during the API call.")
        print(f"   Error details: {e}")
