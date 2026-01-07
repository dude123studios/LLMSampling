import requests
import json
import os
import time
from .base import BaseModel

class APIModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.getenv(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"API Key not found in environment variable: {config.api_key_env}")
            
        self.base_url = config.base_url
        self.model_name = config.model_name
        
    def generate(self, prompt: str, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://neurips-experiment.com", # Required by OpenRouter
            "X-Title": "Sampling Limits NeurIPS",
            "Content-Type": "application/json"
        }
        
        # Default params
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_new_tokens', 4096),
            "top_p": kwargs.get('top_p', 1.0)
        }
        
        retries = 3
        last_error = None
        
        for i in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code != 200:
                    # Log the full error body for debugging
                    print(f"API Error (Attempt {i+1}): {response.status_code} - {response.text}")
                    response.raise_for_status()
                
                resp_json = response.json()
                if 'choices' not in resp_json:
                    print(f"API Error (Unexpected Format): {resp_json}")
                    raise KeyError("'choices' not found in response")
                message = resp_json['choices'][0]['message']
                
                content = message.get('content', '')
                reasoning = message.get('reasoning', '')
                
                # Some models (DeepSeek/Qwen-Math) put chain of thought in 'reasoning'
                if reasoning:
                    combined_output = f"{reasoning}\n\n{content}"
                    return combined_output
                
                if content is None: 
                    content = ""
                    
                return content
                
            except Exception as e:
                print(f"Request failed (Attempt {i+1}): {e}")
                last_error = e
                if i < retries - 1:
                    time.sleep(2 ** i)
                    
        raise last_error or Exception("Unknown API error")
