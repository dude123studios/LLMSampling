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
    """Loads and merges all non-DRYRUN log files."""
    files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No log files found in {results_dir}")
    
    # Filter out DRYRUN
    files = [f for f in files if "DRYRUN" not in f]
    if not files:
        print("Warning: Only DRYRUN logs found. Loading those instead.")
        files = glob.glob(os.path.join(results_dir, "*DRYRUN*.jsonl"))
        
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
        df = load_all_logs(args.results_dir)
        
        plot_similarity_analysis(df, args.output_dir)
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
