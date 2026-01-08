
import argparse
import json
import os
import sys
import yaml
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path hack to ensure we can import mechanistic
sys.path.append(os.getcwd())

# Import only necessary components
from mechanistic.oracles.openrouter import OpenRouterClient
from mechanistic.config import ExperimentConfig
from mechanistic.external.sampling_limits.src.data.loader import load_task_data

class TaskConfig:
    def __init__(self, name="math", dataset="hendrycks/competition_math", split="test", subset_level="Level 5", subset_name=None):
        self.name = name
        self.dataset = dataset
        self.split = split
        self.subset_level = subset_level
        self.subset_name = subset_name

def main():
    parser = argparse.ArgumentParser(description="Generate Oracle Solutions Multithreaded")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--output_file", type=str, default="data/oracle_solutions.json")
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent threads")
    args = parser.parse_args()

    load_dotenv()
    
    # 1. Load Config Manually (Avoid ExperimentRunner init which loads local model)
    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = ExperimentConfig(**config_dict)
    
    # 2. Load Data (Replicating load_data logic without full runner)
    task_name = config.task_name
    print(f"Loading task '{task_name}' data...")
    
    if task_name == "math":
        task_conf = TaskConfig(
            name="math",
            dataset="lighteval/MATH-Hard", 
            split="test",
            subset_level=None
        )
    elif task_name == "gpqa":
        task_conf = TaskConfig(name="gpqa", dataset="Idavidrein/gpqa", split="train", subset_name="gpqa_diamond")
    else:
        # Fallback
        task_conf = TaskConfig(name=task_name, dataset=task_name)

    limit = args.limit if args.limit else config.dataset_limit
    dataset = load_task_data(task_conf, limit=limit, seed=42)
    
    problems = []
    for item in dataset:
         # Standardize prompt/id
         if task_name == "math":
             prompt = item['problem']
             solution = item['solution']
             uid = item.get('unique_id', hash(prompt))
         elif task_name == "gpqa":
             prompt = item['Question']
             solution = item['Correct Answer']
             uid = hash(prompt)
         else:
             prompt = str(item)
             solution = ""
             uid = hash(prompt)
             
         problems.append({
             "id": f"{task_name}_{uid}",
             "prompt": prompt,
             "gold_solution": solution
         })

    print(f"Loaded {len(problems)} problems.")
    
    # 3. Setup Oracle Client (Lightweight)
    oracle = OpenRouterClient(config.oracle_model)
    print(f"Generating oracle solutions using {config.oracle_model} with {args.threads} threads...")
    
    # 4. Load Existing
    solutions = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                solutions = json.load(f)
            print(f"Loaded {len(solutions)} existing solutions.")
        except:
            print("Could not load existing solutions, starting fresh.")
            
    # 5. Multithreaded Execution
    problems_to_solve = [p for p in problems if p['id'] not in solutions]
    print(f"Problems to solve: {len(problems_to_solve)}")
    
    def solve_single(problem):
        try:
            sol = oracle.solve_problem(problem['prompt'])
            return problem['id'], sol
        except Exception as e:
            return problem['id'], None

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # Save progressively
            futures = {executor.submit(solve_single, p): p for p in problems_to_solve}
            
            for future in tqdm(as_completed(futures), total=len(problems_to_solve), desc="Generating"):
                pid, sol = future.result()
                if sol:
                    solutions[pid] = sol
                    
                # Periodic save (every 10 or so? Or just let user ctrl-c)
                # For safety, saving every update might differ on concurrency, 
                # but dict assignment is atomic-ish in GIL. JSON dump is not.
                # Let's save every chunk or just relies on final/interrupt.
                
    except KeyboardInterrupt:
        print("Interrupted by user. Saving...")
        
    # Final Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    print(f"Saved {len(solutions)} solutions to {args.output_file}")

if __name__ == "__main__":
    main()
