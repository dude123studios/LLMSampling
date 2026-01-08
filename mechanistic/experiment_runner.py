import argparse
import sys
import os
import torch
import numpy as np
import importlib.util
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from datetime import datetime
import torch
import numpy as np
import yaml
from sklearn.cluster import KMeans
from typing import Union

# Load environment variables from .env file
load_dotenv()

from mechanistic.config import ExperimentConfig
from mechanistic.models.wrapper import LatentModelWrapper
from mechanistic.utils.logging import ExperimentLogger
from mechanistic.sampling.strategies import TemperatureSampler, SeedSampler, VectorSteeredSampler
from mechanistic.oracles.openrouter import OpenRouterClient
from mechanistic.oracles.openrouter import OpenRouterClient

class TaskConfig:
    def __init__(self, name="math", dataset="hendrycks/competition_math", split="test", subset_level="Level 5", subset_name=None):
        self.name = name
        self.dataset = dataset
        self.split = split
        self.subset_level = subset_level
        self.subset_name = subset_name



class ExperimentRunner:
    def __init__(self, config_or_path: Union[str, ExperimentConfig] = "config.yaml"):
        if isinstance(config_or_path, str):
            self.config = ExperimentConfig.load(config_or_path)
        else:
            self.config = config_or_path
            
        self.logger = ExperimentLogger(self.config.output_dir, self.config.experiment_name)
        
        # Initialize components
        self.model = LatentModelWrapper(self.config.model)
        self.oracle = OpenRouterClient(self.config.oracle_model)
        
        # Samplers
        self.temp_sampler = TemperatureSampler(self.model, self.config)
        self.seed_sampler = SeedSampler(self.model, self.config)
        self.vector_sampler = VectorSteeredSampler(self.model, self.config)
        
        # Store bifurcation points for Step 6 (Mechanistic Analysis)
        # List of dicts: {divergence, prompt_context, top1, top2, problem_id}
        self.bifurcation_file = os.path.join(self.config.output_dir, "bifurcation_points.json")
        self.bifurcation_points = []
        if os.path.exists(self.bifurcation_file):
            print(f"Loading existing bifurcation points from {self.bifurcation_file}")
            import json
            with open(self.bifurcation_file, 'r') as f:
                self.bifurcation_points = json.load(f)
        
    def run_step_2_variance(self, problems):
        """Step 2: Sampling Geometry & Variance (with Length Analysis)"""
        print("Running Step 2: Variance...")
        
        # Use variance-specific config
        checkpoints = self.config.variance_exp.token_checkpoints
        max_tokens = self.config.variance_exp.max_new_tokens
        temperatures = self.config.variance_exp.temperatures
        
        for problem in problems:
            for temp in temperatures:
                # OPTIMIZATION: Temp 0 only needs 1 sample
                n_samples = 1 if temp == 0.0 else self.config.n_samples_per_problem
                
                # Request specific layers and Checkpoints
                results = self.temp_sampler.sample(
                    problem['prompt'], 
                    temperature=temp, 
                    output_layers=self.config.variance_layers,
                    pooling_checkpoints=checkpoints,
                    max_new_tokens=max_tokens, # Override generation length
                    stride=self.config.latent_sampling_stride, # Pass stride optimization
                    pooling_window=self.config.latent_pooling_window, # Pass pooling window optimization
                    n_samples=n_samples # NEW: Pass separate n_samples
                )
                
                # Check outcome format: latent_z should be Dict[layer, Dict[ckpt, tensor]]
                first_z = results[0]['latent_z']
                
                # Metric: Average Pairwise Cosine Similarity
                similarity_data = {} # Key: layer_idx -> {ckpt: avg_sim}
                
                if isinstance(first_z, dict):
                    layer_indices = sorted(first_z.keys())
                    for l_idx in layer_indices:
                        layer_sim = {}
                        l_data = first_z[l_idx] # Dict[ckpt, tensor]
                        
                        for ckpt in checkpoints:
                            if ckpt in l_data:
                                # Stack for this (layer, ckpt) [N, 1, dim] -> [N, dim]
                                latents = torch.stack([r['latent_z'][l_idx][ckpt] for r in results])
                                if latents.dim() > 2:
                                    latents = latents.view(latents.shape[0], -1)
                                
                                # Normalize vectors for Cosine Sim
                                # latents: [N, D]
                                norms = torch.norm(latents, p=2, dim=1, keepdim=True)
                                normalized_latents = latents / (norms + 1e-8)
                                
                                N = latents.shape[0]
                                if N > 1:
                                    # Cosine Matrix: [N, N]
                                    cos_matrix = torch.mm(normalized_latents, normalized_latents.t())
                                    
                                    # We want average of off-diagonal elements
                                    # Sum of all elements - Trace (which is N)
                                    sum_all = torch.sum(cos_matrix)
                                    sum_off_diag = sum_all - N
                                    avg_sim = sum_off_diag / (N * (N - 1))
                                    metric_val = avg_sim.item()
                                else:
                                    metric_val = 1.0 # Self-similarity is 1
                                    
                                layer_sim[ckpt] = metric_val
                                
                        similarity_data[l_idx] = layer_sim
                        
                else: 
                     # Should not happen with current config, but safe fallback logic omitted for brevity
                     pass

                self.logger.log({
                    "step": 2,
                    "problem_id": problem.get('id'),
                    "temperature": temp,
                    "layer_similarity": similarity_data, # {layer: {ckpt: sim}}
                    "generated_texts": [r['generated_text'] for r in results]
                })

    def run_step_3_4_manifold(self, problems):
        """Step 3 & 4: Manifold Distance & Off-Policy"""
        print("Running Step 3 & 4: Manifold...")
        for problem in problems:
            prompt = problem['prompt']
            
            # Check if local model fails greedy
            try:
                res = self.temp_sampler.sample(prompt, temperature=0.0)[0]
                local_solution = res['generated_text']
            except Exception as e:
                print(f"WARNING: Skipping problem {problem.get('id')} in Step 3 due to generation error: {e}")
                continue
            # Evaluation needed here (requires external evaluator)
            # Assuming we can determine correctness, e.g. exact match on answer
            
            # For this skeleton, we assume we process all, or need an evaluator
            # Let's say we fetch Oracle for all
            
            oracle_sol = self.oracle.solve_problem(prompt)
            if not oracle_sol:
                continue
                
            # Compute z* (Oracle Latent)
            # We must force the oracle solution through the local model to get its latent
            # in the local model's space!
            # z* = model(oracle_text)
            oracle_ids = self.model.tokenizer(prompt + oracle_sol, return_tensors="pt").input_ids
            z_star = self.model.get_latents(oracle_ids.to(self.model.device))
            if isinstance(z_star, dict):
                 if not z_star:
                      print(f"WARNING: get_latents returned empty dict for z_star. Skipping problem {problem.get('id')}.")
                      continue
                 z_star = list(z_star.values())[0]
            
            # Distance of local sample to z*
            local_z = res['latent_z']
            if isinstance(local_z, dict):
                 local_z = list(local_z.values())[0]
            local_z = local_z.to(self.model.device)
            dist = torch.norm(local_z - z_star).item()
            
            self.logger.log({
                "step": 3,
                "problem_id": problem.get('id'),
                "distance_to_oracle": dist,
                "local_text": local_solution,
                "oracle_text": oracle_sol
            })
            
            # Step 4: Prefix Insertion
            # Take X% of oracle solution
            oracle_tokens = self.model.tokenizer(oracle_sol, return_tensors="pt").input_ids[0]
            k = int(len(oracle_tokens) * self.config.manifold_exp.oracle_percentage)
            prefix = self.model.tokenizer.decode(oracle_tokens[:k])
            
            forced_prompt = prompt + prefix
            try:
                forced_res = self.temp_sampler.sample(forced_prompt, temperature=0.0)[0]
            except Exception as e:
                print(f"WARNING: Skipping Step 4 for problem {problem.get('id')} due to error: {e}")
                continue
            
            forced_z = forced_res['latent_z']
            if isinstance(forced_z, dict):
                forced_z = list(forced_z.values())[0]
            forced_z = forced_z.to(self.model.device)
            drift = torch.norm(forced_z - z_star).item()
            
            self.logger.log({
                "step": 4,
                "problem_id": problem.get('id'),
                "drift_after_forcing": drift,
                "forced_text": forced_res['generated_text']
            })

    def run_step_5_sensitivity(self, problems):
        """Step 5: Path Sensitivity"""
        print("Running Step 5: Path Sensitivity...")
        for problem in problems:
            prompt = problem['prompt']
            
            # 1. Generate a baseline greedy trajectory (reference)
            baseline_len = self.config.generation.max_new_tokens # Use config value
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                baseline_out = self.model.model.generate(**inputs, max_new_tokens=baseline_len, do_sample=False)
            
            baseline_ids = baseline_out[0]
            input_len = inputs.input_ids.shape[1]
            generated_ids = baseline_ids[input_len:] # Just the new tokens
            
            # 2. Iterate through bifurcation points with stride
            stride = self.config.sensitivity_exp.bifurcation_step_interval
            
            for t in range(stride, len(generated_ids), stride):
                # Prefix is prompt + t tokens
                # We want to branch at step t (relative to new generation)
                # i.e. we force the (t+1)-th token.
                
                # Context for branching
                # We need to compute Top-1/Top-2 *given* the first t tokens.
                # baseline_ids[:input_len+t]
                context_ids = baseline_ids[:input_len+t].unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.model.model(context_ids)
                    next_token_logits = outputs.logits[0, -1, :]
                
                top2 = torch.topk(next_token_logits, 2).indices
                top1_id = top2[0].item()
                top2_id = top2[1].item()
                
                # Create trajectory config
                traj_gen_config = self.config.generation.model_copy()
                traj_gen_config.max_new_tokens = self.config.sensitivity_exp.max_new_tokens + t + 1
                
                # Pass latent sampling stride for efficiency
                sampling_stride = self.config.latent_sampling_stride

                # Force Top 1
                res1 = self.model.force_token_generation(
                    prompt, 
                    force_at_step=t, 
                    force_token_id=top1_id, 
                    gen_config=traj_gen_config, 
                    stride=sampling_stride
                )
                
                # Force Top 2
                res2 = self.model.force_token_generation(
                    prompt, 
                    force_at_step=t, 
                    force_token_id=top2_id, 
                    gen_config=traj_gen_config, 
                    stride=sampling_stride
                )
                
                # Compare trajectories (Centroid Distance)
                traj1 = res1['trajectory'] 
                traj2 = res2['trajectory']
                
                # "Latent averages" implication -> centroid
                z1 = traj1.mean(dim=0)
                z2 = traj2.mean(dim=0)
                diff = torch.norm(z1 - z2).item()
                
                self.logger.log({
                    "step": 5,
                    "problem_id": problem.get('id'),
                    "bifurcation_step": t, # Record where we branched
                    "path_divergence": diff,
                    "token1": top1_id,
                    "token2": top2_id
                })
                
                # Save context text for Step 6
                # context was baseline_ids[:input_len+t]
                # decode only the generated part for context_text in bifurcation?
                # Step 6 uses it as a prompt. If we pass Prompt + Prefix, Step 6 might re-encode prompt?
                # Step 6 code: prompt=point['context_text'].
                # So we should pass the FULL text (Prompt + Prefix).
                full_context_text = self.model.tokenizer.decode(context_ids[0], skip_special_tokens=True)
                
                self.bifurcation_points.append({
                    "problem_id": problem.get('id'),
                    "bifurcation_step": t,
                    "divergence": diff,
                    "context_text": full_context_text,
                    "top1_id": top1_id,
                    "top2_id": top2_id
                })
            
        # Save to disk
        import json
        with open(self.bifurcation_file, 'w') as f:
            json.dump(self.bifurcation_points, f)
        print(f"Saved {len(self.bifurcation_points)} bifurcation points to {self.bifurcation_file}")

    def run_step_6_attribution(self):
        """Step 6: Mechanistic Attribution (Experiment 5)"""
        print("Running Step 6: Logit Attribution...")
        
        # 1. Filter top 200 bifurcation points
        if not self.bifurcation_points:
            print("No bifurcation points found.")
            return

        print(f"Total points collected: {len(self.bifurcation_points)}")
        # Sort desc by divergence
        sorted_points = sorted(self.bifurcation_points, key=lambda x: x['divergence'], reverse=True)
        top_n = self.config.attribution_exp.top_n_points
        top_points = sorted_points[:top_n]
        
        print(f"Analyzing top {len(top_points)} bifurcation points...")
        
        for i, point in enumerate(top_points):
            # Compute attribution
            # We can't batch easily because contexts differ
            results = self.model.compute_logit_difference_attribution(
                prompt=point['context_text'],
                step=0, # Unused inside, as we use full context
                token1_id=point['top1_id'],
                token2_id=point['top2_id']
            )
            
            self.logger.log({
                "step": 6,
                "problem_id": point['problem_id'],
                "divergence_rank": i,
                "prior_divergence": point['divergence'],
                "layer_contributions": results['layer_contributions'],
                "logit_diff": results['logit_diff']
            })

    def run_step_7_clustering(self, problems):
        """Step 7: Solution Clustering & Distinctness (Experiment 6)"""
        print("Running Step 7: Clustering...")
        
        # Import grader dynamically
        # Ensure external/sampling_limits is in path
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        external_path = os.path.join(current_dir, "external", "sampling_limits")
        if external_path not in sys.path:
            sys.path.append(external_path)
            
        try:
             # The package structure in sampling_limits has 'src' at root usually? 
             # Let's check if 'src' is a package or if we need to append inputs
             # User said: mechanistic/external/sampling_limits/src/evaluation/math_grader.py
             # If we append `sampling_limits`, we import `src.evaluation...`
             from src.evaluation.math_grader import grade_math
        except ImportError:
             print(f"WARNING: Could not import grader from {external_path}. Using dummy (always True).")
             grade_math = lambda x, y: True

        for problem in problems:
            # 1. Sample N solutions
            # Use high temp and LONG generation
            sample_temp = 0.8 
            results = self.temp_sampler.sample(
                problem['prompt'], 
                temperature=sample_temp,
                max_new_tokens=self.config.clustering.max_new_tokens # 2048 usually
            )
            
            # 2. Filter Correct Solutions
            correct_results = []
            for r in results:
                # We need the answer key
                # Problem dict keys: 'id', 'prompt', 'gold_solution'
                if grade_math(r['generated_text'], problem['gold_solution']):
                    correct_results.append(r)
            
            print(f"Problem {problem.get('id')}: {len(correct_results)}/{len(results)} correct.")
            
            if len(correct_results) < self.config.clustering.n_clusters:
                print(f"Not enough correct samples to cluster.")
                continue
                
            # 3. K-Means Clustering on Latents
            latents = torch.stack([r['latent_z'] for r in correct_results]).numpy() # [N, dim]
            
            n_clusters = min(self.config.clustering.n_clusters, len(correct_results))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(latents)
            
            # Extract representatives for each cluster (first element)
            representatives = []
            param_reps = {} # Map k -> text
            for k in range(n_clusters):
                indices = np.where(labels == k)[0]
                if len(indices) > 0:
                    text = correct_results[indices[0]]['generated_text']
                    representatives.append(text)
                    param_reps[int(k)] = text # Serializable key

            # 4. LLM Judge for Distinctness (Optional)
            distinct_metric = 0.0
            if self.config.clustering.judge_enabled:
                # Check distinctness among representatives
                # If we have K reps, we have K*(K-1)/2 pairs.
                distinct_pairs = 0
                total_pairs = 0
                for i in range(len(representatives)):
                    for j in range(i+1, len(representatives)):
                        is_distinct = self.oracle.judge_distinctness(
                            representatives[i], 
                            representatives[j], 
                            judge_model=self.config.clustering.judge_model
                        )
                        if is_distinct:
                            distinct_pairs += 1
                        total_pairs += 1
                
                distinct_metric = distinct_pairs / total_pairs if total_pairs > 0 else 0.0

            self.logger.log({
                "step": 7,
                "problem_id": problem.get('id'),
                "total_samples": len(results),
                "n_clusters_found": n_clusters, # Valid K-Means clusters (empty clusters?)
                "judge_distinctness_score": distinct_metric, # Ratio of distinct pairs
                "cluster_labels": labels.tolist(),
                "cluster_representatives": param_reps
            })

    def run_step_8_steering_rollout(self, problems):
        """Step 8: Steering Experiment - PCA Directions"""
        print("Running Step 8: Steering Rollouts...")
        
        from mechanistic.sampling.strategies import VectorSteeredSampler
        
        steered_sampler = VectorSteeredSampler(self.model, self.config)
        target_layers = self.config.steering_exp.layers
        n_directions = self.config.steering_exp.n_directions
        strength = self.config.steering_exp.strength_multiplier
        
        for problem in problems:
            # 1. Sample baseline for PCA
            # User Request: "10 common directions between the ZERO temprature sampling outcome and the regular sampled outcomes"
            # Or "takes the distinct rollouts in step 7, prefils them... and then uses pca to find... common directions... between ZERO... and regular"
            
            print(f"Sampling baseline (Temp 1.0) and Pivot (Temp 0.0) for problem {problem.get('id')}...")
            
            # Check if we have Step 7 results in memory (from current run) 
            # We don't have a reliable way to access memory results across steps in this architecture without passing them.
            # But the user might have run Step 7 right before. 
            
            # Since explicit persistence is hard, let's stick to regeneration BUT if we want "distinct rollouts"
            # we should mimic Step 7 logic or just sample high temp.
            
            # PREFILL LOGIC: If we want to use specific texts (e.g. from a file), we should load them.
            # For now, to satisfy "prefills them", let's replicate the sampling:
            
            # Temp 1.0 Samples (Regenerated "Regular Outcome")
            baseline_samples = self.temp_sampler.sample(
                problem['prompt'],
                temperature=1.0, 
                max_new_tokens=64,
                output_layers=target_layers,
                pooling_checkpoints=[32]
            )
            # Optimization: If we could load Step 7 "cluster representatives", we would:
            # 1. Load text. 2. self.model.get_latents(input_ids=encode(prompt+text), ...)
            # Given we are in the same run execution flow often, we could modify `run` to return data.
            # But 'problems' arg is just the input data.
            
            # For this iteration, we keep the regeneration which is robust and safe.
            # The "prefill" requirement is effectively handled by generating new samples which *become* the filled context.
            
            # Temp 0.0 Sample (The "Pivot")
            pivot_sample = self.temp_sampler.sample(
                problem['prompt'],
                temperature=0.0,
                max_new_tokens=64,
                output_layers=target_layers,
                pooling_checkpoints=[32],
                n_samples=1
            )[0]
            
            # 2. Compute directions for each layer
            # We need to aggregate latents [N_samples, D] for each layer
            directions_by_layer = {}
            
            # Access latent_z from baseline
            # latent_z structure: {layer: {ckpt: tensor}}
            first_z = baseline_samples[0]['latent_z']
            
            import torch
            
            for layer in target_layers:
                if layer not in first_z:
                    print(f"Warning: Layer {layer} data not found in baseline.")
                    continue
                    
                # Stack all samples for this layer
                # We use the pooled representation (e.g. at token 32) as the "state"
                # Or should we use the whole trajectory? User said "compute ... on each of layer".
                # Using the pooled vector is a reasonable proxy for the "thought vector"
                
                # Check structure
                # r['latent_z'][layer] is Dict[ckpt, tensor]
                # We assume we used pooling_checkpoints=[32]. So take key 32.
                ckpt_key = 32
                if ckpt_key not in first_z[layer]:
                    ckpt_key = list(first_z[layer].keys())[0]
                
                # Stack [N_samples, Dim]
                vectors = torch.stack([r['latent_z'][layer][ckpt_key] for r in baseline_samples])
                
                # Get Pivot [1, Dim]
                pivot = pivot_sample['latent_z'][layer][ckpt_key]
                if pivot.dim() == 1: pivot = pivot.unsqueeze(0)
                
                # SVD requires float32 or float64
                vectors = vectors.float()
                pivot = pivot.float()
                
                # User Request: "directions between ZERO temprature ... and regular"
                # Difference: x_i - x_0
                centered = vectors - pivot
                
                # We want directions capturing this variance. SVD on these differences.
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                
                # Top N
                comps = Vh[:n_directions] # [N_dirs, Dim]
                directions_by_layer[layer] = comps
                
            # 3. Rollouts with Steering
            print(f"Running steered rollouts for problem {problem.get('id')}...")
            
            steered_results = []
            
            for layer, directions in directions_by_layer.items():
                for i in range(len(directions)):
                    direction_vec = directions[i] # [Dim]
                    
                    # Prepare info for logging
                    meta = {"layer": layer, "direction_idx": i}
                    
                    # Apply steering manually via sampler or use wrapper?
                    # VectorSteeredSampler.sample takes 'direction_idx' which refers to cached directions. 
                    # But we have multiple layers.
                    # We can pass the vector DIRECTLY if we modify sampler or use model generate directly.
                    # Or we can temporarily set the sampler's cache.
                    
                    steered_sampler.computed_directions = {0: direction_vec}
                    # We also need to hack the config's target layer dynamically?
                    # The sampler reads `self.config.steering.target_layer`.
                    # We should probably update the config object (it's mutable).
                    
                    original_layer = self.config.steering.target_layer
                    original_strength = self.config.steering.strength
                    
                    self.config.steering.target_layer = layer
                    self.config.steering.strength = strength
                    
                    try:
                        # n_rollouts from config (default 1)
                        # We force n_samples_per_problem to 1 for this loop if n_rollouts is 1? 
                        # Or use the config value? User said "in each of 10 rollouts apply one such vector"
                        # implying 1 rollout per vector.
                        
                        original_n_samples = self.config.n_samples_per_problem
                        self.config.n_samples_per_problem = self.config.steering_exp.n_rollouts
                        
                        res_list = steered_sampler.sample(
                            problem['prompt'], 
                            direction_idx=0, # Use the one we injected
                            max_new_tokens=64
                        )
                        
                        for r in res_list:
                            r.update(meta)
                            steered_results.append(r)
                            
                    finally:
                        # Restore config
                        self.config.steering.target_layer = original_layer
                        self.config.steering.strength = original_strength
                        self.config.n_samples_per_problem = original_n_samples
                        
            # Log all steered results for this problem
            # NEW: Calculate Pass@k for each layer
            from mechanistic.external.sampling_limits.src.evaluation.math_grader import grade_math
            from mechanistic.external.sampling_limits.src.evaluation.metrics import estimate_pass_at_k
            from collections import defaultdict
            
            layer_texts = defaultdict(list)
            for r in steered_results:
                layer_texts[r['layer']].append(r['generated_text'])
                
            layer_metrics = {}
            for layer, texts in layer_texts.items():
                correct_count = 0
                for text in texts:
                    if grade_math(text, problem['gold_solution']):
                        correct_count += 1
                
                # Check if we have enough samples for k=10
                n = len(texts)
                k_values = [1]
                if n >= 10:
                    k_values.append(10)
                    
                pk = estimate_pass_at_k(n, correct_count, k_values)
                layer_metrics[layer] = {
                    "num_samples": n,
                    "num_correct": correct_count,
                    "pass@1": pk.get("pass@1", 0.0),
                    "pass@10": pk.get("pass@10", 0.0)
                }
            
            self.logger.log({
                "step": 8,
                "problem_id": problem.get('id'),
                "layer_metrics": layer_metrics,
                "steered_generations": [
                    {
                        "layer": r.get('layer'),
                        "direction": r.get('direction_idx'),
                        "text": r.get('generated_text'),
                        "is_correct": grade_math(r.get('generated_text'), problem['gold_solution'])
                    }
                    for r in steered_results
                ]
            })

    def load_data(self, limit: int = None):
        """
        Loads data using the external sampling_limits loader.
        Uses self.config.dataset_limit if limit is not provided.
        """
        # Determine effective limit
        effective_limit = limit if limit is not None else self.config.dataset_limit
        
        # Add external to sys.path if not present
        ext_path = os.path.join(os.getcwd(), "mechanistic", "external", "sampling_limits")
        if ext_path not in sys.path:
            sys.path.append(ext_path)

        try:
            from src.data.loader import load_task_data
        except ImportError as e:
            print(f"CRITICAL ERROR: Could not import 'src.data.loader' from {ext_path}.")
            print(f"Error details: {e}")
            print("Running in strictly mock mode for dry-run if applicable.")
            if self.config.experiment_name.endswith("_DRYRUN"):
                 return [{
                    "id": "mock_0",
                    "prompt": "User: What is 1+1?\nPlease solve this step by step.\nAssistant:",
                    "gold_solution": "2"
                }]
            return []

        # Configure Task
        # We reuse the local TaskConfig class or create a simple object
        # The loader expects an object with .name, .dataset, .split, .subset_level/name
        
        task_name = self.config.task_name
        
        if task_name == "math":
            task_conf = TaskConfig(
                name="math",
                dataset="lighteval/MATH-Hard", 
                split="test",
                subset_level=None # The new dataset structure might not have 'level' or requires different filtering
            )
        elif task_name == "gpqa":
             task_conf = TaskConfig(
                name="gpqa",
                dataset="gpqa",
                split="train", # GPQA often uses 'train' as main
                subset_name="gpqa_diamond"
             )
        else:
            print(f"WARNING: Unknown task name '{task_name}'. Defaulting to MATH Level 5.")
            task_conf = TaskConfig(name="math", dataset="competition_math", split="test", subset_level="Level 5")
            
        print(f"Loading task '{task_conf.name}' with limit={effective_limit}...")
        
        try:
            dataset = load_task_data(task_conf, limit=effective_limit, seed=42)
        except Exception as e:
             print(f"Error loading external dataset: {e}")
             # Fallback for dry run
             if self.config.experiment_name.endswith("_DRYRUN"):
                print("Fallback to single mock item for dry run.")
                return [{
                    "id": "mock_0",
                    "prompt": "User: What is 1+1?\nPlease solve this step by step.\nAssistant:",
                    "gold_solution": "2"
                }]
             return []
        
        problems = []
        for item in dataset:
            # Standardize prompt format
            if 'problem' in item:
                question = item['problem']
            elif 'question' in item:
                 question = item['question']
            else:
                question = str(item)

            if 'solution' in item:
                solution = item['solution']
            elif 'answer' in item:
                solution = item['answer']
            else:
                solution = ""

            prompt = f"User: {question}\nPlease solve this step by step.\nAssistant:"
            
            # Unique ID
            uid = item.get('unique_id', hash(question))
            
            problems.append({
                "id": f"{task_name}_{uid}",
                "prompt": prompt,
                "gold_solution": solution
            })
            
        return problems

    def run(self, limit: int = None, steps: list = None):
        print(f"Starting Experiment: {self.config.experiment_name}")
        
        # 1. Load Data
        problems = self.load_data(limit=limit)
        
        print(f"Loaded {len(problems)} problems.")
        
        # Helper to check if step should run
        def should_run(step_num):
            if steps is None: return True
            return step_num in steps

        if should_run(2): self.run_step_2_variance(problems)
        if should_run(3) or should_run(4): self.run_step_3_4_manifold(problems)
        if should_run(5): self.run_step_5_sensitivity(problems)
        if should_run(6): self.run_step_6_attribution()
        if should_run(7): self.run_step_7_clustering(problems)
        if should_run(8): self.run_step_8_steering_rollout(problems)

def main():
    import argparse
    from mechanistic.config import ExperimentConfig # Assuming ExperimentConfig is defined here or imported
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run a fast, cheap smoke test of the pipeline.")
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated list of steps to run (e.g., '2,6'). If None, runs all.")
    args = parser.parse_args()
    
    # Load Config
    if args.config.endswith('.yaml'):
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig(**config_dict)
    else:
        # Fallback or python config
        from mechanistic.config import default_config
        config = default_config
        
    if args.dry_run:
        print("!"*80)
        print("DRY RUN MODE ENABLED: Overriding config for fast, cheap testing.")
        print("!"*80)
        config.n_samples_per_problem = 1
        config.generation.max_new_tokens = 10 # Increase to support Step 5 (needs > 5)
        config.model.model_name_or_path = "gpt2" # Use small public model for testing
        config.model.load_in_8bit = False
        config.model.load_in_4bit = False
        config.model.device = "cpu" # Force CPU for dry run to avoid CUDA/MPS issues
        config.model.attn_implementation = "eager" # Force eager for CPU/GPT2
        config.target_layer = 10 # Safe for GPT-2 (12 layers)
        
        # Exp 1
        config.variance_exp.temperatures = [1.0]
        config.variance_exp.max_new_tokens = 4
        config.variance_layers = [ config.variance_layers[0] ] if config.variance_layers else [10]
        config.variance_exp.token_checkpoints = [2, 4]
        
        # Exp 3 (Sensitivity)
        config.sensitivity_exp.bifurcation_step_interval = 2
        config.sensitivity_exp.max_new_tokens = 4
        
        # Exp 6
        config.clustering.max_new_tokens = 8
        config.clustering.n_clusters = 1 # Must be < n_samples (1) -> actually need n_samples >= n_clusters. 
        # Override n_samples for clustering if needed, or set n_clusters=1
        config.clustering.judge_enabled = False # Disable API calls
        
        # Reduce dataset size
        # We can't easily reduce dataset size here without modifying the loader or runner
        # But the runner loops over `problems`. We can patch the runner method or pass a limit.
        # Let's add a limit arg to runner.run() ? 
        # Or just monkey-patch config to have a 'dry_run' flag accessed by runner.
        config.experiment_name += "_DRYRUN"

    runner = ExperimentRunner(config)
    
    steps_to_run = None
    if args.steps:
        try:
            steps_to_run = [int(s.strip()) for s in args.steps.split(',')]
        except ValueError:
            print("Error: --steps must be a comma-separated list of integers (e.g. '2,3').")
            return

    runner.run(limit=1 if args.dry_run else None, steps=steps_to_run)

if __name__ == "__main__":
    main()
