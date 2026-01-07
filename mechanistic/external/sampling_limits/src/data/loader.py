import random
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def get_subset(dataset, limit: int, seed: int):
    """
    Returns a deterministic subset of the dataset.
    CRITICAL: Must correspond to the same indices across different runs.
    """
    if limit is None or limit >= len(dataset):
        return dataset
    
    # Create a stable random generator
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    
    selected_indices = indices[:limit]
    # HF Datasets .select() reorders based on input list, so we sort for safety
    return dataset.select(sorted(selected_indices))

def load_task_data(task_config, limit=None, seed=42):
    """
    Loads and preprocesses data for a specific task.
    """
    logger.info(f"Loading task: {task_config.name}")
    
    if task_config.name == "math":
        ds = load_dataset(task_config.dataset, split=task_config.split)
        # Filter for Level 5 (if specified)
        if task_config.subset_level is not None:
             ds = ds.filter(lambda x: x['level'] == task_config.subset_level)
    elif task_config.name == "gpqa":
        ds = load_dataset(task_config.dataset, task_config.subset_name, split=task_config.split)
    elif task_config.name == "code":
        ds = load_dataset(task_config.dataset, split=task_config.split)
    else:
        raise ValueError(f"Unknown task: {task_config.name}")
        
    return get_subset(ds, limit, seed)
