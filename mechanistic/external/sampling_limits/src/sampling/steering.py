import torch

class SteeringContext:
    def __init__(self, model, layer_idx, vector, coeff):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = vector
        self.coeff = coeff
        self.hook_handle = None
        
        # Resolve the actual layer module
        # This assumes Llama-like architecture: model.model.layers
        if hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
             self.layer = model.model.model.layers[layer_idx]
        elif hasattr(model.model, "layers"):
             self.layer = model.model.layers[layer_idx]
        else:
             raise ValueError("Could not locate layers in model architecture")

    def _hook_fn(self, module, args, output):
        # output is tuple(hidden_states, ...)
        # hidden_states shape: [batch, seq, dim]
        hidden_states = output[0]
        
        # Determine strict compatibility of device and dtype
        vec = self.vector.to(hidden_states.device).to(hidden_states.dtype)
        
        # Add vector to the LAST token position only
        # We use inplace addition if safe, or create new tensor
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + (self.coeff * vec)
        
        return (hidden_states,) + output[1:]

    def __enter__(self):
        self.hook_handle = self.layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle:
            self.hook_handle.remove()
