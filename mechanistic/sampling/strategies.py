import torch
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np

from mechanistic.config import ExperimentConfig, GenerationConfig
from mechanistic.models.wrapper import LatentModelWrapper

class BaseSampler(ABC):
    def __init__(self, model: LatentModelWrapper, config: ExperimentConfig):
        self.model = model
        self.config = config
        
    @abstractmethod
    def sample(self, prompt: str, **kwargs) -> List[Dict]:
        """Generate samples and return results with latents."""
        pass

class TemperatureSampler(BaseSampler):
    """
    Samples at various temperatures.
    """
    def sample(self, prompt: str, temperature: float = 1.0, **kwargs) -> List[Dict]:
        config = self.config.generation.model_copy()
        config.temperature = temperature
        # Ensure sampling is on if temp > 0
        config.do_sample = True if temperature > 0 else False
        
        return self.model.generate(
            [prompt] * self.config.n_samples_per_problem,
            config,
            **kwargs
        )

class SeedSampler(BaseSampler):
    """
    Samples with explicit seeds to ensure reproducibility or explore specific random perturbations.
    """
    def sample(self, prompt: str, seeds: List[int]) -> List[Dict]:
        results = []
        base_config = self.config.generation.model_copy()
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            # Generate one by one to respect the seed
            res = self.model.generate([prompt], base_config)[0]
            res["seed"] = seed
            results.append(res)
            
        return results

class VectorSteeredSampler(BaseSampler):
    """
    Adds a steering vector to residual streams during generation.
    Requires pre-computing PCA or directions.
    """
    def __init__(self, model: LatentModelWrapper, config: ExperimentConfig):
        super().__init__(model, config)
        self.computed_directions = {} # CACHE

    def compute_directions(self, latents: torch.Tensor, n_components: int = 3):
        """
        Compute PCA components from a batch of latents.
        latents: [N, dim]
        """
        # Simple PCA
        # Center the data
        mean = torch.mean(latents, dim=0)
        centered = latents - mean
        
        # SVD
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Components are rows of Vh
        components = Vh[:n_components]
        
        self.computed_directions = {i: components[i] for i in range(n_components)}
        return self.computed_directions

    def sample(self, prompt: str, direction_idx: int = 0, sign: float = 1.0, **kwargs) -> List[Dict]:
        """
        Sample with +strength * direction added to residual stream.
        """
        if direction_idx not in self.computed_directions:
            raise ValueError(f"Direction {direction_idx} not computed yet.")
            
        direction = self.computed_directions[direction_idx]
        # Allow strength override if user modified config
        strength = self.config.steering.strength * sign
        
        steering_vector = direction * strength
        
        # Determine target layer (usually mid-to-late layers are effective)
        target_layer = self.config.steering.target_layer
        steering_dict = {target_layer: steering_vector}
        
        return self.model.generate(
            [prompt] * self.config.n_samples_per_problem,
            self.config.generation,
            steering_vectors=steering_dict,
            **kwargs
        )
