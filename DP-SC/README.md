# DP-QoSC

DP-QoSC is a differential privacy experimental project for QoS-aware service composition. It includes candidate pool construction, differentially private selection, parameter experiments, and user preference inference attacks.

## 1. Environment Setup

Python 3.9 or 3.10 is recommended. Run the following commands from the project root directory.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## 2. Required Parameter Settings

The differential privacy and candidate-pool parameters are defined in `dp_para` in `src/dp_perturb.py`, including:

```python
epsilon                 # Privacy budget
clip_bounds             # Utility clipping threshold
optimal_candidate_count # Number of high-quality candidates
pipeline_runs           # Number of independent runs, default: 20
```

The representative candidate-pool size and the maximum utility degradation bound can also be configured.

## 3. Running the Code

### 3.1 Generate Reference Solutions

Before running the main experiment, generate the reference solutions for the selected dataset:

```bash
python src/Benchmark.py [dataset]
```

### 3.2 Run the Main DP-QoSC Experiment

```bash
python src/OnDemand_GA.py [dataset]
```

The experimental results are saved to:

```text
src/results/<dataset>_dp_qosc_results.txt
```

### 3.3 Run Candidate-Pool Parameter Ablation Experiments

```bash
python src/OnDemand_GA.py [dataset] ablation
```

### 3.4 Run the Inference Attack Experiment

```bash
python src/Inference_Attack_PoolSize_Representative.py [dataset]
```

To reduce the number of tasks or runs, use:

```text
--task-limit N
--runs N
--workers N
```

### 3.5 Run the ClipBound/OptCount Inference Attack Ablation Experiment

```bash
python src/Inference_Attack_ClipBound_OptCount_Ablation.py [dataset]
```
