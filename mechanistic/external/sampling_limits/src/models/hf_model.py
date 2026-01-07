import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class HFModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.device = config.get("device_map", "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=self.device,
            torch_dtype=getattr(torch, config.torch_dtype, torch.float16)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def generate(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt based on length approximately or just return full and handle in downstream
        return decoded[len(prompt):] 
