# The Geometry of Sampling

This repository implements experiments to analyze the geometric structure of Large Language Model (LLM) latent spaces during reasoning tasks. Specifically, it investigates how sampling temperature impacts the variance of latent representations and how diverse solutions cluster in high-dimensional space.

## Experiments Overview

## Experiments Overview

### Experiment 1: Sampling Variance vs. Layer
**Objective:** quantify the "uncertainty" of the model's internal representations across layers and generation lengths.
**Methodology:**
1.  **Latent Representation ($z$):** Defined as the strided mean-pooling of the residual stream at a specific layer $l$ over the generated tokens from $t=1$ to $T_{gen}$. To optimize memory and computational efficiency, we sample every $k$-th token (defined by `latent_sampling_stride`, default $k=32$).
    $$z = \frac{1}{|S|} \sum_{t \in S} h_{l}^{(t)}, \quad S = \{k, 2k, 3k, \dots\} \cap [1, T_{gen}]$$
2.  **Variance Metric:** For a given prompt $x$ and temperature $T$, we sample $N$ completions. We collect the set of latent vectors $\{z_1, \dots, z_N\}$. The variance is defined as the trace of the covariance matrix of these vectors:
    $$Var(T) = \text{Tr}(\text{Cov}(\{z_i\}))$$
3.  **Checkpoints:** Variance is computed at multiple generation lengths (e.g., 128, 256, 512, 1024 tokens) to observe if the model "commits" to a path early or late.
4.  **Layer Analysis:** We sweep across layers to identify where "creativity" or "uncertainty" peaks.

### Experiment 2: Manifold Analysis (Steps 3 & 4)
**Objective:** Compare the geometric trajectory of model-generated solutions against an "Oracle" (usually a stronger model or ground truth) trajectory.
**Methodology:**
1.  **Oracle Trajectory:** We compute the latent representation $z^*$ of an oracle solution (e.g., from DeepSeek R1).
2.  **Manifold Distance:** We measure the Euclidean distance between a locally sampled solution's latent $z_{local}$ and the oracle latent:
    $$\text{Dist} = ||z_{local} - z^*||_2$$
3.  **Off-Policy Drift (Prefix Forcing):** We take the first $P\%$ (default 30%) of the oracle solution tokens and force them as a prefix for the local model. We then measure the distance between the resulting latent $z_{forced}$ and $z^*$. This quantifies how much the model "drifts" even when guided.

### Experiment 3: Path Sensitivity (Step 5)
**Objective:** Measure the divergence in model trajectories when forced to choose the second-most likely token ($w_{top2}$) instead of the greedy choice ($w_{top1}$) at various points in the solution.
**Methodology:**
1.  **Baseline Generation:** Generate a complete greedy solution trajectory $T_{baseline}$.
2.  **Iterative Bifurcation:** Iterate through the baseline at intervals of $k$ tokens (defined by `bifurcation_step_interval`, default 32).
3.  **Forced Branching:** At each interval $t$, create a branch where we force the model to select its second-most likely token ($w_{top2}$) instead of the baseline token ($w_{top1}$).
4.  **Trajectory Comparison:** Decode greedily for $M$ tokens (default 64) for both branches and measure the Euclidean distance between their strided mean-pooled latents:
    $$D(t) = ||z_{baseline}[t:t+M] - z_{forced}[t:t+M]||_2$$


### Experiment 4: Mechanistic Attribution (Step 6)
**Objective:** Attribute the cause of the path divergence to specific layers.
**Methodology:**
1.  **Bifurcation Points:** Utilize the high-divergence points identified in Experiment 3. This step automatically loads `bifurcation_points.json` if previously generated.
2.  **Logit Difference Attribution:** We approximate the contribution of each layer to the logit difference between top-1 and top-2 tokens:
    $$\text{Contrib}_l = (W_U[w_{top1}] - W_U[w_{top2}])^T (h_l - h_{l-1})$$
    This assumes a linear decomposition of the residual stream.

### Experiment 6: Solution Clustering & Distinctness (Step 7)
**Objective:** Determine if high-temperature sampling produces distinct semantic modes (clusters) of solutions.
**Methodology:**
1.  **Sampling:** Generate $N$ solutions at high temperature (e.g., $T=0.8$) with a long context window.
2.  **Filtering:** Filter solutions for mathematical correctness using regular expression matching on the boxed answer.
3.  **Clustering:** Apply K-Means clustering on the latent representations ($z$) of the correct solutions.
4.  **Distinctness Evaluation:** An LLM Judge compares representative solutions from different clusters to determine if they are biologically distinct methods.


## Setup & Installation

1.  **Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **External Dependencies:**
    This project relies on the `sampling_limits` evaluation suite. Ensure the `mechanistic/external/sampling_limits` directory is populated.

3.  **API Keys:**
    Create a `.env` file for OpenRouter (used for the LLM Judge):
    ```bash
    OPENROUTER_API_KEY=your_key_here
    ```

## Usage

### Running Experiments

The `experiment_runner.py` script orchestrates the pipeline. You can run all experiments sequentially or run specific experiments individually using the `--steps` flag.

#### 1. Run Everything
Execute the entire pipeline (Steps 2-7):
```bash
python3 -m mechanistic.experiment_runner --config config.yaml
```

#### 2. Run Individual Experiments
You can isolate specific experiments using the `--steps` argument (comma-separated list of step IDs).

**Experiment 1: Sampling Variance (Step 2)**
Calculates latent variance across layers and generation lengths.
```bash
python3 -m mechanistic.experiment_runner --config config.yaml --steps 2
```

**Experiment 2: Manifold Analysis (Steps 3 & 4)**
Computes distance to oracle solution and off-policy drift.
```bash
python3 -m mechanistic.experiment_runner --config config.yaml --steps 3,4
```

**Experiment 3: Path Sensitivity (Step 5)**
Measures divergence when forcing top-1 vs top-2 tokens.
```bash
python3 -m mechanistic.experiment_runner --config config.yaml --steps 5
```

**Experiment 4: Mechanistic Attribution (Step 6)**
Attributes divergence to specific layers (requires data from Step 5).
```bash
python3 -m mechanistic.experiment_runner --config config.yaml --steps 6
```

**Experiment 6: Solution Clustering (Step 7)**
Clusters solutions and evaluates distinctness with an LLM Judge.
```bash
python3 -m mechanistic.experiment_runner --config config.yaml --steps 7
```

### Dry Run (Verification)
To verify the configuration and pipeline logic without incurring API costs or waiting for long generations, use the `--dry-run` flag. This uses mock data, a small model (GPT-2), and minimal generation parameters.

```bash
python3 -m mechanistic.experiment_runner --config config.yaml --dry-run
# Or specific steps in dry-run mode
python3 -m mechanistic.experiment_runner --config config.yaml --dry-run --steps 2
```
**Use this check before every full run to ensure your config is valid.**

### Visualization
After running experiments, generate plots:

```bash
python3 mechanistic/analysis/visualize.py
```
Outputs will be saved to `experiments/plots/`.

## Configuration (`config.yaml`)

The configuration is hierarchical. Key sections:

*   **`model`**:
    *   `model_name_or_path`: Local HF path or ID (e.g., `Qwen/Qwen2.5-8B-Instruct`).
    *   `device`: `cuda` (or `mps` for Mac).
*   **`variance_exp` (Exp 1)**:
    *   `temperatures`: List of temps to sweep (e.g., `[0.0, 1.0]`).
    *   `token_checkpoints`: List of lengths to pool over (e.g., `[128, 256]`).
    *   `max_new_tokens`: Limit for this specific experiment.
    *   Note: `variance_layers` is defined in the root config.
*   **`manifold_exp` (Exp 2)**:
    *   `oracle_percentage`: Fraction of oracle solution to use as prefix (e.g., 0.3).
*   **`sensitivity_exp` (Exp 3)**:
    *   `bifurcation_step_interval`: Interval $k$ to branch at (e.g., 32).
    *   `lookahead_tokens`: Tokens to generate to determine top-2 probabilities.
    *   `max_new_tokens`: Length of trajectory to generate after forcing.
*   **`attribution_exp` (Exp 4)**:
    *   `top_n_points`: Number of bifurcation points to analyze.
*   **`clustering` (Exp 6)**:
    *   `enabled`: Toggle this step.
    *   `n_clusters`: Target number of K-Means clusters.
    *   `judge_model`: OpenRouter model ID to use for distinctness evaluation (e.g., `deepseek/deepseek-r1-distill-qwen-32b`).
    *   `judge_enabled`: Enable LLM Judge (requires `OPENROUTER_API_KEY`).
    *   `max_new_tokens`: Generates longer sequences (2048) to ensure solution completion.

## Code Structure

*   `mechanistic/experiment_runner.py`: Orchestrates the pipeline.
*   `mechanistic/models/wrapper.py`: `LatentModelWrapper` handles specific architecture logic (GPT2, Qwen, Llama) and latent extraction.
*   `mechanistic/config.py`: Pydantic models for type-safe configuration.
