import math
from scipy.special import comb

def estimate_pass_at_k(num_samples, num_correct, k_values):
    """
    Estimates pass@k using the unbiased estimator from the HumanEval paper.
    
    Formula: pass@k = 1 - (combinations(n-c, k) / combinations(n, k))
    where n = num_samples, c = num_correct
    
    Args:
        num_samples (int): Total number of samples generated (n).
        num_correct (int): Number of correct samples (c).
        k_values (list[int]): List of k values to estimate pass@k for.
        
    Returns:
        dict: Mapping from k to pass@k score.
    """
    results = {}
    n = num_samples
    c = num_correct
    
    for k in k_values:
        if k > n:
            results[f"pass@{k}"] = 0.0 # Cannot estimate if k > n
            continue
            
        if c == 0:
            results[f"pass@{k}"] = 0.0
            continue
            
        # pass@k = 1 - binom(n-c, k) / binom(n, k)
        # We use scipy.special.comb for stability with large numbers, 
        # though n is typically small (<100) in these experiments.
        
        try:
            val = 1.0 - (comb(n - c, k) / comb(n, k))
            results[f"pass@{k}"] = float(val)
        except Exception:
            results[f"pass@{k}"] = 0.0
            
    return results
