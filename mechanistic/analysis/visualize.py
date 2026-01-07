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

def load_latest_log(results_dir):
    files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No log files found in {results_dir}")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading log file: {latest_file}")
    
    data = []
    with open(latest_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
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

def plot_layer_similarity_vs_layer(df, output_dir):
    """Step 2: Plot Avg Cosine Similarity vs Layer (for different Temps)"""
    step2_df = df[df['step'] == 2]
    if step2_df.empty:
        print("No Step 2 data found.")
        return

    # Data 'layer_similarity' column is now a dict {layer: {ckpt: sim}}
    
    records = []
    for _, row in step2_df.iterrows():
        sim_data = row.get('layer_similarity', row.get('variance', {})) # Backward compat
        temp = row['temperature']
        
        if isinstance(sim_data, dict):
            for layer, ckpt_data in sim_data.items():
                if layer == "target": continue
                
                # Checkpoints: 32, 64, 128
                if isinstance(ckpt_data, dict):
                    for ckpt, val in ckpt_data.items():
                        records.append({
                            "Temperature": temp,
                            "Layer": int(layer),
                            "Similarity": val,
                            "Checkpoint": int(ckpt)
                        })
                else: 
                     # Fallback
                     records.append({
                        "Temperature": temp,
                        "Layer": int(layer),
                        "Similarity": ckpt_data,
                        "Checkpoint": "Full"
                    })
            
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("No similarity data to plot for Step 2.")
        return

    # Plot Similarity vs Layer (Factored by Checkpoint)
    # We want a plot where X=Layer, Y=Similarity, Hue=Temp.
    # User requested: "graph of layer variance given by layer... 32 deep, 64 deep..."
    # We can facet by Checkpoint or hue by checkpoint? 
    # Let's do separate plots or subplots for each checkpoint to clearly see the bound.
    
    checkpoints = sorted(list(set(plot_df['Checkpoint'])))
    for ckpt in checkpoints:
        ckpt_df = plot_df[plot_df['Checkpoint'] == ckpt]
        
        plt.figure(figsize=(10, 6))
        
        if "Layer" in ckpt_df.columns:
            sns.lineplot(
                data=ckpt_df,
                x='Layer',
                y='Similarity',
                hue='Temperature',
                palette="viridis",
                linewidth=2,
                marker='o'
            )
            plt.title(f"Layer Similarity vs Layer (Gen Length: {ckpt})")
            plt.xlabel("Layer Index")
            plt.ylabel("Avg Cosine Similarity")
            plt.ylim(0, 1.05) # Cosine similarity bound
            
            path = os.path.join(output_dir, f"similarity_vs_layer_ckpt_{ckpt}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {path}")

def plot_mani_dist(df, output_dir):
    """Step 3/4: Plot Distance to Correct Manifold"""
    step3_df = df[df['step'] == 3]
    if step3_df.empty:
        print("No Step 3 data found.")
        return

    plt.figure(figsize=(8, 6))
    
    # Histogram style
    sns.histplot(
        data=step3_df, 
        x='distance_to_oracle', 
        bins=20, 
        kde=True,
        color=COLORS['data_point'],
        edgecolor='black',
        linewidth=1,
        line_kws={'color': COLORS['fit_line'], 'linewidth': 2} # KDE line style
    )
    
    plt.title("Dist to Oracle Manifold")
    plt.xlabel("L2 Dist to z*")
    
    plt.savefig(os.path.join(output_dir, "manifold_distance_dist.png"), bbox_inches='tight', dpi=300)
    print(f"Saved plot to {os.path.join(output_dir, 'manifold_distance_dist.png')}")

def plot_drift(df, output_dir):
    """Step 4: Drift after forcing"""
    step4_df = df[df['step'] == 4]
    if step4_df.empty:
        return
        
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=step4_df, 
        x='drift_after_forcing', 
        bins=20, 
        kde=True, 
        color=COLORS['fit_line'], # Use orange for drift/forcing to distinguish
        edgecolor='black',
        linewidth=1
    )
    plt.title("Latent Drift After Forcing")
    plt.xlabel("Drift from Oracle Traj")
    
    plt.savefig(os.path.join(output_dir, "prefix_forcing_drift.png"), bbox_inches='tight', dpi=300)
    print(f"Saved plot to {os.path.join(output_dir, 'prefix_forcing_drift.png')}")

def plot_attribution(df, output_dir):
    """Step 6: Logit Difference Attribution"""
    step6_df = df[df['step'] == 6]
    if step6_df.empty:
        return
        
    # Data is 'layer_contributions': list of floats
    # Expand into rows for seaborn
    # We want to aggregate across all samples to show the "average circuit"
    
    # Create long-form DF
    records = []
    for _, row in step6_df.iterrows():
        contribs = row['layer_contributions']
        for i, val in enumerate(contribs):
            label = "Embed" if i == 0 else f"L{i-1}"
            records.append({
                "Layer": label,
                "LayerIdx": i,
                "Contribution": val
            })
            
    plot_df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 6))
    
    # Boxplot to show distribution of contributions
    sns.boxplot(
        data=plot_df,
        x='Layer',
        y='Contribution',
        color=COLORS['data_point'], # Blue boxes
        boxprops={'edgecolor': 'black', 'linewidth': 1},
        medianprops={'color': COLORS['fit_line'], 'linewidth': 2},
        showfliers=False # Hide outliers for cleanliness, or True to see extremes
    )
    
    # Add a horizontal line at 0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    plt.title("Layer Attribution to Top-1 vs Top-2 Logit Diff")
    plt.xlabel("Layer")
    plt.ylabel("contrib to logit(y1) - logit(y2)")
    plt.xticks(rotation=45)
    
    # Save
    path = os.path.join(output_dir, "logit_attribution.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {path}")

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
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        set_style()
        df = load_latest_log(args.results_dir)
        
        plot_layer_similarity_vs_layer(df, args.output_dir)
        plot_mani_dist(df, args.output_dir)
        plot_drift(df, args.output_dir)
        plot_attribution(df, args.output_dir)
        plot_clustering_distinctness(df, args.output_dir)
        
    except Exception as e:
        print(f"Error extracting plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
