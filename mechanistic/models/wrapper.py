import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Dict, Callable, Union, Any
import contextlib

from mechanistic.config import ModelConfig, GenerationConfig

class LatentModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        
        print(f"Loading model: {config.model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
        
        # Robust Attention Config
        attn_impl = config.attn_implementation
        if attn_impl == "flash_attention_2":
            try:
                 import flash_attn
            except ImportError:
                 print("WARNING: flash_attn package not found. Falling back from 'flash_attention_2' to 'eager' attention.")
                 attn_impl = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=getattr(torch, config.dtype),
            trust_remote_code=config.trust_remote_code,
            device_map=self.device,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            attn_implementation=attn_impl,
        )
        self.model.eval()
        
        # Identify layers - Qwen/Llama usually have model.layers
        # GPT2 has transformer.h
        # OPT has model.decoder.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            self.layers = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
             self.layers = self.model.transformer.h
        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
             self.layers = self.model.model.decoder.layers
        else:
            raise ValueError(f"Could not find layers in model structure: {type(self.model)}")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Ensure model knows about it if it's new
            # resize_token_embeddings? Not mostly needed if we reuse eos
            
        self.num_layers = len(self.layers)
        
    def _extract_latents_hook(self, module, input, output, layer_idx: int, storage: dict, stride: int = 1):
        """Hook to extract residual stream output from a layer."""
        # Residual stream is usually output[0] in HF transformers
        if isinstance(output, tuple):
            hidden_state = output[0] 
        else:
            hidden_state = output
            
        # Store only the last token's hidden state to save memory during generation
        # shape: (batch_size, seq_len, hidden_dim)
        # Apply stride
        if stride > 1 and hidden_state.shape[1] > 1:
             # Take every Nth token
             # We might miss the last one if we just slice.
             # Ideally we want the last one too?
             # User said "average those ones".
             # Let's slice [:, ::stride, :]
             hidden_state = hidden_state[:, ::stride, :]
             
        storage[layer_idx] = hidden_state.detach()

    def _steering_hook(self, module, input, output, steering_vector: torch.Tensor, layer_idx: int):
        """Hook to add steering vector to residual stream."""
        if isinstance(output, tuple):
            hidden_state = output[0]
            rest = output[1:]
        else:
            hidden_state = output
            rest = ()
            
        # Add steering vector
        # Vector shape should broadcast to (batch, seq, hidden)
        # Usually vector is (1, 1, hidden) or (1, hidden)
        hidden_state = hidden_state + steering_vector.to(hidden_state.device).to(hidden_state.dtype)
        
        if isinstance(output, tuple):
            return (hidden_state,) + rest
        else:
            return hidden_state

    @contextlib.contextmanager
    def capture_latents(self, start_layer_fraction: float = 2/3, stride: int = 1):
        """Context manager to capture residual streams from the last fraction of layers."""
        start_layer = int(self.num_layers * start_layer_fraction)
        handles = []
        storage = {}
        
        try:
            for i in range(start_layer, self.num_layers):
                hook = lambda m, inp, out, idx=i, store=storage: self._extract_latents_hook(m, inp, out, idx, store, stride=stride)
                handles.append(self.layers[i].register_forward_hook(hook))
            yield storage
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def steer(self, steering_vectors: Dict[int, torch.Tensor]):
        """Context manager to apply steering vectors to specific layers."""
        handles = []
        try:
            for layer_idx, vector in steering_vectors.items():
                if 0 <= layer_idx < self.num_layers:
                    hook = lambda m, inp, out, vec=vector, idx=layer_idx: self._steering_hook(m, inp, out, vec, idx)
                    handles.append(self.layers[layer_idx].register_forward_hook(hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def get_latents(self, 
                   input_ids: torch.Tensor, 
                   prompt_length: int = 0, 
                   layers: Union[List[int], str] = "target", # "all", "target", or list
                   target_layer_idx: int = None, # If None, use config
                   pooling_checkpoints: Optional[List[int]] = None, # List of token counts to pool over
                   stride: int = 1,
                   pooling_window: Optional[int] = None # Number of strides to look back
                   ) -> Union[torch.Tensor, Dict[int, Union[torch.Tensor, Dict[int, torch.Tensor]]]]:
        """
        Extracts latent representations.
        
        Args:
            input_ids: Full sequence input ids.
            prompt_length: Index to start pooling from (to ignore prompt). 
                           Latents = Mean(Layer[batch, prompt_length:, :])
            layers: "all" for all layers, "target" for single config layer, or list of ints.
        """
        if target_layer_idx is None:
            target_layer_idx = self.config.target_layer if hasattr(self.config, 'target_layer') else 22
            
        # Determine layers to process
        layer_indices = []
        if layers == "all":
            layer_indices = list(range(self.num_layers))
        elif layers == "target":
            # Clamp index to valid range
            if target_layer_idx >= self.num_layers:
                target_layer_idx = self.num_layers - 1
            layer_indices = [target_layer_idx]
        else:
            layer_indices = layers
            
        storage = {}
        
        # Use strided capture
        with torch.no_grad():
            handles = []
            for idx in layer_indices:
                 if 0 <= idx < self.num_layers:
                    def hook(module, input, output, i=idx, store=storage, s=stride):
                         if isinstance(output, tuple): h = output[0]
                         else: h = output
                         h = h.detach() 
                         if s > 1 and h.shape[1] > 1:
                             h = h[:, ::s, :]
                         store[i] = h
                    handles.append(self.layers[idx].register_forward_hook(hook))
            
            try:
                self.model(input_ids, output_hidden_states=False) 
            finally:
                for h in handles: h.remove()
            
        results = {}
        
        import math
        
        for l_idx in layer_indices:
            if l_idx not in storage:
                continue
                
            state = storage[l_idx] 
            
            # Start pooling from prompt. Prompt is also strided.
            strided_prompt_len = math.ceil(prompt_length / stride)
            
            if pooling_checkpoints:
                layer_results = {}
                
                for ckpt in pooling_checkpoints:
                    original_end = prompt_length + ckpt
                    strided_end = math.ceil(original_end / stride)
                    strided_end = min(strided_end, state.shape[1])
                    
                    if strided_end > strided_prompt_len:
                        start_idx = strided_prompt_len
                        if pooling_window:
                            # If window is set, only pool over the last N strides
                            start_idx = max(strided_prompt_len, strided_end - pooling_window)
                            
                        pooled = state[:, start_idx:strided_end, :].mean(dim=1)
                    else:
                        pooled = state[:, -1, :] 
                        
                    layer_results[ckpt] = pooled
                
                results[l_idx] = layer_results
            else:
                if strided_prompt_len < state.shape[1]:
                    pooled = state[:, strided_prompt_len:, :].mean(dim=1) 
                else:
                    pooled = state[:, -1, :]
                
                results[l_idx] = pooled
            
        # Return single tensor if only target requested (legacy compatibility)
        if layers == "target" and len(results) == 1:
            return list(results.values())[0]
            
        return results

    def generate(self, 
                 prompts: List[str], 
                 gen_config: GenerationConfig, 
                 steering_vectors: Optional[Dict[int, torch.Tensor]] = None,
                 output_layers: Union[str, List[int]] = "target",
                 pooling_checkpoints: Optional[List[int]] = None,
                 **kwargs) -> List[Dict]:
        """
        Generate completions, optionally with steering.
        Returns list of dicts with text and final latent vector z.
        kwargs can override gen_config values (e.g. max_new_tokens).
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        # Prepare context managers
        ctx_stack = contextlib.ExitStack()
        
        # Add latent capture context
        # We only really care about the latent of the FINAL token generated, 
        # but capture all to be safe or if we need trajectory.
        # For now, efficient implementation would be complex.
        # Simplest: Generate first, then run one forward pass on full sequence to get latents.
        
        # Steering needs to be active *during* generation
        if steering_vectors:
            ctx_stack.enter_context(self.steer(steering_vectors))

        with ctx_stack:
            with torch.no_grad():
                # Allow kwargs to override config
                gen_kwargs = {
                    "max_new_tokens": kwargs.get("max_new_tokens", gen_config.max_new_tokens),
                    "do_sample": kwargs.get("do_sample", gen_config.do_sample),
                    "temperature": kwargs.get("temperature", gen_config.temperature),
                    "top_p": kwargs.get("top_p", gen_config.top_p),
                    "top_k": kwargs.get("top_k", gen_config.top_k),
                    "repetition_penalty": kwargs.get("repetition_penalty", gen_config.repetition_penalty),
                    "pad_token_id": self.tokenizer.pad_token_id
                }
                print(f"DEBUG GENERATE: input_len={input_len}, max_new_tokens={gen_kwargs['max_new_tokens']}, full_kwargs={gen_kwargs}", flush=True)
                
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
        
        # Decode
        results = []
        for i, output_ids in enumerate(outputs):
            full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            new_text = self.tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
            
            # Post-hoc latent extraction on the FULL generated sequence
            # to get the stable representation of the final answer
            # We pass prompt_length to pool only generated tokens
            stride = int(kwargs.get("stride", 1))
            pooling_window = kwargs.get("pooling_window")
            
            latent_z_or_dict = self.get_latents(
                output_ids.unsqueeze(0), 
                prompt_length=input_len,
                layers=output_layers,
                pooling_checkpoints=pooling_checkpoints,
                stride=stride,
                pooling_window=pooling_window
            ) # [1, dim] or Dict[int, Tensor]
            
            # If it's a dict, keep it as dict. If tensor, keep tensor.
            # But the sampler expects `latent_z` key.
            # We can store the WHOLE dict in 'latent_z' key (it's just a python obj)
            # But verify if we need to cpu().
            
            if isinstance(latent_z_or_dict, dict):
                latent_z_cpu = {}
                for k, v in latent_z_or_dict.items():
                    if isinstance(v, dict):
                        latent_z_cpu[k] = {ck: cv.cpu() for ck, cv in v.items()}
                    else:
                        latent_z_cpu[k] = v.cpu()
            else:
                latent_z_cpu = latent_z_or_dict.cpu()

            results.append({
                "full_text": full_text,
                "generated_text": new_text,
                "latent_z": latent_z_cpu, 
                "tokens": output_ids.cpu()
            })
            
        return results

    def force_token_generation(self, 
                             prompt: str, 
                             force_at_step: int, 
                             force_token_id: int,
                             gen_config: GenerationConfig,
                             stride: int = 1) -> Dict:
        """
        Forces a specific token at `force_at_step` (relative to new tokens) and generates greedily thereafter.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # 1. Generate up to step t
        # If force_at_step is 0, we force the FIRST generated token.
        
        current_ids = input_ids
        
        # Check if we need to generate prefix first
        if force_at_step > 0:
            with torch.no_grad():
                prefix_out = self.model.generate(
                    **inputs,
                    max_new_tokens=force_at_step,
                    do_sample=gen_config.do_sample, # Use config for prefix
                    temperature=gen_config.temperature
                )
            current_ids = prefix_out # [1, seq_len + force_at_step]
            
        # 2. Force the token
        forced_token_tensor = torch.tensor([[force_token_id]], device=self.device)
        current_ids = torch.cat([current_ids, forced_token_tensor], dim=1)
        
        # 3. Generate the rest GREEDILY (as per Step 5 requirements usually)
        # or use the config? The plan says "From that point on, decode greedily"
        greedy_config = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens - force_at_step - 1,
            do_sample=False, # Greedy
            temperature=1.0
        )
        
        # We need to capture latents for the REST of the trajectory
        # to measure divergence.
        
        # We can use the existing generate method for the tail, 
        # but we need to pass the FULL context (current_ids) as input_ids 
        # BUT generate() usually takes "prompts". 
        # Better to access model.generate directly with input_ids
        
        with torch.no_grad():
            # We want to capture latents here.
            # But capture_latents captures ALL forward passes.
            # model.generate calls forward many times.
            # Storing all of them is heavy.
            # For "trajectory", we usually mean the sequence of states.
            
            # For simplicity in this experiment:
            # Generate the full text first.
            # Then run ONE forward pass on the full sequence to get all hidden states.
            
            tail_out = self.model.generate(
                input_ids=current_ids,
                max_new_tokens=greedy_config.max_new_tokens,
                do_sample=False
            )
            
        final_ids = tail_out
        
        # 4. Extract Trajectory Latents
        # We want z_t for all t > force_at_step
        # Run forward pass on full sequence
        with torch.no_grad():
            with self.capture_latents(start_layer_fraction=2/3, stride=stride) as storage:
                self.model(final_ids, output_hidden_states=True)
                
        # Storage has {layer: [1, seq_len, dim]}
        # We want to aggregate layers (mean pool) -> [seq_len, dim]
        layer_latents = []
        for idx in sorted(storage.keys()):
            layer_latents.append(storage[idx][0]) # [seq_len, dim]
            
        stacked = torch.stack(layer_latents) # [n_layers, seq_len, dim]
        trajectory = torch.mean(stacked, dim=0) # [seq_len, dim]
        
        full_text = self.tokenizer.decode(final_ids[0], skip_special_tokens=True)
        
        return {
            "full_text": full_text,
            "trajectory": trajectory,
            "generated_text": full_text 
        }
        
    def compute_logit_difference_attribution(self, 
                                           prompt: str, 
                                           step: int, 
                                           token1_id: int, 
                                           token2_id: int) -> Dict[str, Any]:
        """
        Computes the contribution of each layer to the logit difference (logit(tok1) - logit(tok2)).
        Uses the linear decomposition: logits = Wu * (h_emb + sum(h_layer_i)).
        Contribution_i = (Wu[tok1] - Wu[tok2]) @ (h_layer_i - h_layer_{i-1}).
        """
        # 1. Re-generate to the exact state at 'step'
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # If step is generating the t-th token, we need context up to t-1
        # But 'step' usually means the step index in the *generation*.
        # Let's assume input_ids is the FULL context including generation up to immediately before the bifurcation.
        
        # Since the call site in runner constructs `prompt` (which might be original prompt),
        # we need to be careful. The runner should pass the *text up to that point*.
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            
        # hidden_states is tuple: (embeds, layer_1, ..., layer_N)
        # All have shape (batch, seq, dim)
        hidden_states = outputs.hidden_states
        
        # We care about the LAST position (the one predicting the next token)
        final_position_states = [h[:, -1, :] for h in hidden_states] # List of (batch, dim)
        
        # 2. Get Unembedding Weights for the two tokens
        # self.model.lm_head points to the linear layer
        # weight shape: (vocab, dim)
        w_u = self.model.lm_head.weight # (vocab, dim)
        v = w_u[token1_id] - w_u[token2_id] # (dim,)
        v = v.detach()
        
        # 3. Compute Layer Contributions
        # Contrib 0 is embeddings: hidden_states[0]
        # Contrib i is layer i: hidden_states[i] - hidden_states[i-1]
        
        contributions = []
        
        # Embeddings contribution
        h_emb = final_position_states[0] # (batch, dim)
        # Project onto v
        # (batch, dim) @ (dim,) -> (batch,)
        c_emb = torch.matmul(h_emb, v).item()
        contributions.append(c_emb)
        
        # Layer contributions
        for i in range(1, len(final_position_states)):
            h_curr = final_position_states[i]
            h_prev = final_position_states[i-1]
            h_diff = h_curr - h_prev # The output of block i-1 (since index is shifted by 1 for embeds)
            
            c_layer = torch.matmul(h_diff, v).item()
            contributions.append(c_layer)
            
        return {
            "layer_contributions": contributions, # Index 0 is embed, 1..N are layers
            "logit_diff": sum(contributions) + (self.model.lm_head.bias[token1_id] - self.model.lm_head.bias[token2_id]).item() if self.model.lm_head.bias is not None else sum(contributions)
        }

