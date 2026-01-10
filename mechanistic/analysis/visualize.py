import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os
import matplotlib as mpl

# --- STYLE CONFIGURATION ---
COLORS = {
    "data_point": "#4E79A7",
    "fit_line": "#F28E2B",
    "ci_band": "#F28E2B"
}

def set_style():
    """Applies strict ICML styling requirements."""
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 1.0
    plt.rcParams['axes.axisbelow'] = True # Grid behind plot elements
    
    # Optional: cleaner look
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def load_all_logs(results_dir):
    """Loads and merges all non-DRYRUN log files recursively."""
    # Search recursively for .jsonl files
    files = glob.glob(os.path.join(results_dir, "**/*.jsonl"), recursive=True)
    if not files:
        # Fallback to non-recursive if glob fails or just in case
        files = glob.glob(os.path.join(results_dir, "*.jsonl"))
        
    if not files:
        raise FileNotFoundError(f"No log files found in {results_dir}")
    
    # Filter out DRYRUN
    files = [f for f in files if "DRYRUN" not in f]
    if not files:
        print("Warning: Only DRYRUN logs found. Loading those instead.")
        files = glob.glob(os.path.join(results_dir, "**/*DRYRUN*.jsonl"), recursive=True)
        
    print(f"Loading {len(files)} log files...")
    
    data = []
    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not data:
        return pd.DataFrame()
        
    return pd.DataFrame(data)

def abbreviate_label(text):
    """Abbreviate common terms for axis labels."""
    mapping = {
        "Temperature": "Temp",
        "Variance": "Var",
        "Distance": "Dist",
        "Oracle": "Oracle",
        "Trajectory": "Traj"
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def plot_similarity_analysis(df, output_dir):
    """Step 2: Plot Avg Cosine Similarity vs Temperature (Faceted by Layer, Hue by Depth)"""
    step2_df = df[df['step'] == 2]
    if step2_df.empty:
        print("No Step 2 data found.")
        return

    records = []
    for _, row in step2_df.iterrows():
        sim_data = row.get('layer_similarity', row.get('variance', {}))
        temp = row['temperature']
        
        if isinstance(sim_data, dict):
            for layer, ckpt_data in sim_data.items():
                if layer == "target": continue
                
                if isinstance(ckpt_data, dict):
                    for ckpt, val in ckpt_data.items():
                        records.append({
                            "Temperature": temp,
                            "Layer": int(layer),
                            "Similarity": val,
                            "Depth": int(ckpt)
                        })
                else: 
                     records.append({
                        "Temperature": temp,
                        "Layer": int(layer),
                        "Similarity": ckpt_data,
                        "Depth": "Full"
                    })
            
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("No similarity data to plot.")
        return

    # User Request: "side by side graphs for each layer ... variance based on temperature at all depths"
    # X=Temp, Y=Similarity, Hue=Depth, Col=Layer
    
    unique_layers = sorted(plot_df['Layer'].unique())
    num_layers = len(unique_layers)
    
    # Setup Subplots (wrap cols)
    cols = 4 # Adjust based on number of layers
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), sharey=True)
    if rows == 1 and cols == 1: axes = [axes]
    axes = axes.flatten() if num_layers > 1 else [axes]
    
    for i, layer in enumerate(unique_layers):
        ax = axes[i]
        layer_df = plot_df[plot_df['Layer'] == layer]
        
        sns.lineplot(
            data=layer_df,
            x='Temperature',
            y='Similarity',
            hue='Depth',
            palette="viridis",
            marker='o',
            ax=ax
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Temperature" if i >= (rows-1)*cols else "")
        ax.set_ylabel("Avg Cosine Similarity" if i % cols == 0 else "")
        if i != 0: ax.legend_.remove() # Only show legend on first/last or distinct?
        
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_vs_temp_by_layer.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {path}")


def plot_attribution(df, output_dir):
    """Step 6: Logit Difference Attribution (Bar Plot via User Request)"""
    step6_df = df[df['step'] == 6]
    if step6_df.empty:
        return
        
    records = []
    for _, row in step6_df.iterrows():
        contribs = row['layer_contributions'] # List[float]
        for i, val in enumerate(contribs):
            label = "Embed" if i == 0 else f"L{i-1}"
            records.append({
                "Layer Name": label,
                "Layer Index": i,
                "Contribution": val
            })
            
    plot_df = pd.DataFrame(records)
    
    # User Request: "bar graph for each layer that goes pos or negative"
    # We aggregate by layer -> Mean + Std/CI
    
    plt.figure(figsize=(14, 6))
    
    sns.barplot(
        data=plot_df,
        x='Layer Name',
        y='Contribution',
        color=COLORS['data_point'],
        edgecolor='black',
        errorbar=('ci', 95) # Show confidence intervals
    )
    
    plt.axhline(0, color='black', linewidth=1)
    
    plt.title("Mean Layer Attribution to Top-1 vs Top-2 Logit Difference")
    plt.xlabel("Layer")
    plt.ylabel("Avg Logit Diff Contribution")
    plt.xticks(rotation=45)
    
    path = os.path.join(output_dir, "logit_attribution_bar.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {path}")


def plot_manifold_analysis(df, output_dir):
    """Step 3/4: Manifold Distance & Forcing Impact"""
    step3_df = df[df['step'] == 3].copy()
    step4_df = df[df['step'] == 4].copy()
    
    if step3_df.empty and step4_df.empty:
        print("No Step 3/4 data found.")
        return

    # 1. Dist Distribution
    if not step3_df.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=step3_df, 
            x='distance_to_oracle', 
            bins=20, 
            kde=True,
            color=COLORS['data_point'],
            edgecolor='black',
            linewidth=1,
            line_kws={'color': COLORS['fit_line'], 'linewidth': 2}
        )
        plt.title("L2 Distance: Local solution latents vs Oracle latent")
        plt.xlabel("L2 Distance")
        plt.savefig(os.path.join(output_dir, "manifold_distance_dist.png"), bbox_inches='tight', dpi=300)
        print(f"Saved plot to {os.path.join(output_dir, 'manifold_distance_dist.png')}")

    # 2. Forcing Effectiveness (Scatter: Initial Drift vs Forced Drift)
    # We need to join Step 3 (Initial) and Step 4 (Forced) on problem_id
    if not step3_df.empty and not step4_df.empty:
        merged = pd.merge(
            step3_df[['problem_id', 'distance_to_oracle']], 
            step4_df[['problem_id', 'drift_after_forcing']],
            on='problem_id',
            how='inner'
        )
        
        if not merged.empty:
            plt.figure(figsize=(8, 8))
            
            # Scatter
            sns.scatterplot(
                data=merged,
                x='distance_to_oracle',
                y='drift_after_forcing',
                color=COLORS['data_point'],
                s=100,
                edgecolor='black',
                alpha=0.7
            )
            
            # 1:1 Line (No improvement region)
            max_val = max(merged['distance_to_oracle'].max(), merged['drift_after_forcing'].max())
            plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='No Improvement (y=x)')
            
            plt.title("Effect of Prefix Forcing (Oracle Injection)")
            plt.xlabel("Original Drift (Distance to Oracle)")
            plt.ylabel("Drift After Forcing 30% Oracle")
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, "forcing_impact_scatter.png"), bbox_inches='tight', dpi=300)
            print(f"Saved plot to {os.path.join(output_dir, 'forcing_impact_scatter.png')}")
            
            # 3. Improvement Distribution (Bar Chart)
            # Calculate Improvement = Original - Forced
            # Positive = Improvement (Metric Decreased)
            merged['improvement'] = merged['distance_to_oracle'] - merged['drift_after_forcing']
            merged = merged.sort_values('improvement', ascending=True) # Sort for visual clarity
            merged['color'] = merged['improvement'].apply(lambda x: '#2ca02c' if x > 0 else '#d62728') # Green if positive, Red if negative
            
            plt.figure(figsize=(10, 6))
            
            # Since strict Bar Graph might be messy with many IDs, we use index as x-axis
            plt.bar(
                range(len(merged)), 
                merged['improvement'], 
                color=merged['color'],
                edgecolor='none',
                width=1.0
            )
            
            plt.axhline(0, color='black', linewidth=1)
            plt.title("Improvement in Latent Drift after Forcing 30% Oracle")
            plt.xlabel("Problem Instance (Sorted by Improvement)")
            plt.ylabel("Reduction in Drift (Original - Forced)")
            
            # Optional: Add text stats
            pos_pct = (merged['improvement'] > 0).mean() * 100
            plt.text(0.05, 0.95, f"{pos_pct:.1f}% Improved", transform=plt.gca().transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(os.path.join(output_dir, "forcing_improvement_bar.png"), bbox_inches='tight', dpi=300)
            print(f"Saved plot to {os.path.join(output_dir, 'forcing_improvement_bar.png')}")

def plot_clustering_distinctness(df, output_dir):
    """Step 7: Clustering & Distinctness"""
    step7_df = df[df['step'] == 7]
    if step7_df.empty:
        return
        
    # Plot histogram of Distinctness Scores
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=step7_df,
        x='judge_distinctness_score',
        bins=10,
        kde=False,
        color=COLORS['data_point'],
        edgecolor='black'
    )
    plt.title("Distribution of Solution Distinctness (LLM Judge)")
    plt.xlabel("Pairwise Distinctness Ratio")
    plt.xlim(0, 1)
    
    path = os.path.join(output_dir, "solution_distinctness.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results")
    parser.add_argument("--output_dir", default="experiments/plots")
    parser.add_argument("--file", default=None, help="Specific file to plot (overrides results_dir search)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        set_style()
        
        if args.file:
             print(f"Loading specific file: {args.file}")
             data = []
             with open(args.file, 'r') as f:
                 for line in f:
                     if line.strip(): data.append(json.loads(line))
             df = pd.DataFrame(data)
        else:
             df = load_all_logs(args.results_dir)
        
        plot_similarity_analysis(df, args.output_dir)
        # Replaced separate plots with unified one
        plot_manifold_analysis(df, args.output_dir) 
        plot_attribution(df, args.output_dir)
        plot_clustering_distinctness(df, args.output_dir)
    
    except Exception as e:
        print(f"Error extracting plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
