import yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

class ModelConfig(BaseModel):
    model_name_or_path: str = "Qwen/Qwen2.5-8B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: str = "eager" # Use "flash_attention_2" for speed if supported

class GenerationConfig(BaseModel):
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    
class SteeringConfig(BaseModel):
    enabled: bool = False
    method: str = "pca" # pca, mean_diff
    source_layer: int = 20 # Layer to extract steering vectors from (if applicable)
    target_layer: int = 20 # Layer to apply steering
    strength: float = 1.0
    n_directions: int = 1 # Number of PCA components to use

class VarianceExpConfig(BaseModel):
    temperatures: List[float] = [0.0, 1.0]
    token_checkpoints: List[int] = [128, 256, 512, 1024]
    max_new_tokens: int = 1024

class ClusteringExpConfig(BaseModel):
    enabled: bool = True
    n_clusters: int = 5 
    judge_model: str = "deepseek/deepseek-r1-distill-qwen-32b"
    judge_enabled: bool = False
    max_new_tokens: int = 2048 # Longer generation for full solution checking

class ManifoldExpConfig(BaseModel):
    enabled: bool = True
    oracle_percentage: float = 0.3 # Percentage of oracle solution to use as prefix
    
class SensitivityExpConfig(BaseModel):
    enabled: bool = True
    bifurcation_step_interval: int = 32 # Step interval to check bifurcation (e.g. 32, 64, 96...)
    lookahead_tokens: int = 5 # Number of tokens to generate to find top-2
    max_new_tokens: int = 64 # Length of trajectory to generate after forcing

class AttributionExpConfig(BaseModel):
    enabled: bool = True
    top_n_points: int = 200 # Number of bifurcation points to analyze
    
class ExperimentConfig(BaseModel):
    experiment_name: str
    output_dir: str = "experiments/results"
    data_path: str = "data/raw" # Path to where user puts external data
    
    # Task Config
    task_name: str = "math" # Task name for external loader (e.g. "math", "gpqa")
    dataset_limit: Optional[int] = None # Limit samples from dataset
    
    # Sampling parameters
    n_samples_per_problem: int = 10
    temperatures: List[float] = Field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0])
    temperatures: List[float] = Field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0])
    seeds: Optional[List[int]] = None
    latent_sampling_stride: int = 32 # Sample every N tokens to save memory
    
    # Model & Generation
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # Exp 1 (Variance)
    variance_exp: VarianceExpConfig = Field(default_factory=VarianceExpConfig)

    # Exp 2 (Manifold)
    manifold_exp: ManifoldExpConfig = Field(default_factory=ManifoldExpConfig)

    # Exp 3 (Sensitivity)
    sensitivity_exp: SensitivityExpConfig = Field(default_factory=SensitivityExpConfig)

    # Exp 4 (Attribution)
    attribution_exp: AttributionExpConfig = Field(default_factory=AttributionExpConfig)
    
    # For Step 2 & 5
    steering: SteeringConfig = Field(default_factory=SteeringConfig)
    
    # For Step 7 (Clustering)
    clustering: ClusteringExpConfig = Field(default_factory=ClusteringExpConfig)
    
    # Deprecated fields (moved to sub-configs, keeping for compatibility if needed or removed)
    # temperatures: List[float] = [0.0, 1.0] 
    
    # For Step 3 & 4
    oracle_model: str = "deepseek/deepseek-r1" # OpenRouter ID
    
    # Latent Definition
    target_layer: int = 22 # Layer to use for "the" latent representation (Exp 2-7)
    # Exp 1 Layer Selection
    variance_layers: List[int] = Field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
