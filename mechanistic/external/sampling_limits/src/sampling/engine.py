import torch
import random
from .steering import SteeringContext
from .pruning import SemanticPruner

def run_sampling(model, prompt, method_config):
    """
    Dispatcher for different sampling methods.
    Returns: (output, prompt_used)
    """
    
    # helper to inject seed if needed
    prompt_used = prompt
    if method_config.get('use_seed_prompt', False):
        seed_val = 2
        prompt_used = f"[ATTEMPT PATHWAY #{seed_val}]: {prompt} \n"

    # --- A. Baseline & Seed ---
    if method_config.name == "baseline":
        return model.generate(prompt_used, **method_config), prompt_used
    
    # --- B. Plan-then-Think ---
    elif method_config.name == "plan_then_think":
        stage1_prompt = f"{prompt_used}\n\n[Instruction]: List 3 distinct high-level plans to solve this problem. Do not solve it yet."
        plans_output = model.generate(stage1_prompt, max_new_tokens=512, **method_config)
        stage2_prompt = f"{stage1_prompt}\n{plans_output}\n\n[Instruction]: Now, using the plans above, solve the problem step-by-step."
        return model.generate(stage2_prompt, **method_config), stage2_prompt

    # --- C. Injection / Early Pruning ---     
    elif method_config.name == "injection":
        current_prompt = prompt_used
        full_generation = ""
        depth = method_config.get('depth', 1)
        for d in range(depth + 1):
             output = model.generate(current_prompt, max_new_tokens=method_config.max_new_tokens_chunk, **method_config)
             full_generation += output
             if d < depth:
                 injection = method_config.injection_token
                 full_generation += injection
                 current_prompt = prompt_used + full_generation 
             
        return full_generation, current_prompt # Return the final prompt used for the last chunk
        
    elif method_config.name == "pruning":
        pruner = SemanticPruner(method_config)
        current_prompt = prompt_used
        full_output = ""
        max_chunks = method_config.max_chunks
        
        for _ in range(max_chunks):
             chunk = model.generate(current_prompt, max_new_tokens=method_config.chunk_size, **method_config)
             if pruner.is_redundant(chunk):
                 full_output += chunk + " [PRUNED] Instead,"
                 current_prompt = prompt_used + full_output
                 pruner.add_failure(chunk)
             else:
                 full_output += chunk
                 current_prompt = prompt_used + full_output
             if len(chunk) < 10: break
                 
        return full_output, current_prompt
    
    # --- D. Vector Steering ---         
    elif method_config.name == "vector_steering":
        vector = torch.load(method_config.vector_source)
        with SteeringContext(model, method_config.layer_idx, vector, method_config.coeff):
             return model.generate(prompt_used, **method_config), prompt_used

    # --- E. LoRA Randomness ---
    elif method_config.name == "lora_randomness":
        if hasattr(model, "apply_lora_noise"):
            with model.apply_lora_noise(method_config.noise_alpha):
                return model.generate(prompt_used, **method_config), prompt_used
        else:
            return model.generate(prompt_used, **method_config), prompt_used
            
    else:
        raise ValueError(f"Unknown sampling method: {method_config.name}")
