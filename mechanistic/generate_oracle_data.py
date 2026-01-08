
import argparse
import json
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# Path hack to ensure we can import mechanistic
sys.path.append(os.getcwd())

from mechanistic.experiment_runner import ExperimentRunner
from mechanistic.oracles.openrouter import OpenRouterClient

def main():
    parser = argparse.ArgumentParser(description="Generate Oracle Solutions")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--output_file", type=str, default="data/oracle_solutions.json")
    args = parser.parse_args()

    load_dotenv()
    
    # Init Runner just to load data and config
    runner = ExperimentRunner(args.config)
    problems = runner.load_data(limit=args.limit)
    
    print(f"Loaded {len(problems)} problems.")
    print(f"Generating oracle solutions using {runner.config.oracle_model}...")
    
    solutions = {}
    
    # Check for existing
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                solutions = json.load(f)
            print(f"Loaded {len(solutions)} existing solutions.")
        except:
            print("Could not load existing solutions, starting fresh.")
            
    # Iterate
    try:
        for p in tqdm(problems):
            pid = p['id']
            if pid in solutions:
                continue
                
            prompt = p['prompt']
            sol = runner.oracle.solve_problem(prompt)
            
            if sol:
                solutions[pid] = sol
                # Auto-save
                with open(args.output_file, 'w') as f:
                    json.dump(solutions, f, indent=2)
                    
    except KeyboardInterrupt:
        print("Interrupted by user. Saving...")
        
    with open(args.output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    print(f"Saved {len(solutions)} solutions to {args.output_file}")

if __name__ == "__main__":
    main()
