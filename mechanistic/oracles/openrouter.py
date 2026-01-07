import os
import requests
import json
from typing import Optional, Dict

class OpenRouterClient:
    def __init__(self, model_id: str = "deepseek/deepseek-r1"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_id = model_id
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY not found in environment. Oracle calls will fail.")

    def solve_problem(self, problem_text: str) -> Optional[str]:
        """
        Get the golden solution for a problem.
        """
        if not self.api_key:
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mechanistic-experiments.local", 
        }
        
        data = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": f"Solve this math problem step-by-step:\n\n{problem_text}"}
            ]
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return None
            
        return None

    def judge_distinctness(self, sol1: str, sol2: str, judge_model: str = None) -> bool:
        """
        Asks the judge model if two solutions are fundamentally distinct in their reasoning.
        Returns True if distinct, False otherwise.
        """
        if not self.api_key:
            return False
            
        model = judge_model if judge_model else self.model_id
        
        prompt = f"""Compare the following two math solutions. 
        Focus on the *reasoning approach* and *method*.
        
        Solution 1:
        {sol1}
        
        Solution 2:
        {sol2}
        
        Are these solutions using fundamentally different reasoning paths or methods? 
        Minor wording differences or distinct intermediate arithmetic do not count as distinct methods.
        
        Answer with JSON: {{"distinct": boolean, "reason": "string"}}
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mechanistic-experiments.local", 
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return parsed.get("distinct", False)
                
        except Exception as e:
            print(f"Error calling Judge: {e}")
            return False # Conservative default
            
        return False
