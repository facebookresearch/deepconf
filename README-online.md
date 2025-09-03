# DeepConf (online version)

DeepConf is a confidence-based approach for improving model performance while reducing computational costs. This method selects high-confidence responses to achieve better accuracy with significant token savings compared to traditional self-consistency baselines.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for vLLM)
- Git

### Installation

1. **Install Dynasor for judgement**
   ```bash
   git clone https://github.com/hao-ai-lab/Dynasor.git
   cd Dynasor && pip install . && cd -
   ```

2. **Install Python Dependencies**
   ```bash
   pip install transformers numpy pandas tqdm
   ```

3. **Install vLLM (Custom Version)**
   ```bash
   git clone https://github.com/Viol2000/vllm.git
   cd vllm && git checkout conf-stop
   VLLM_USE_PRECOMPILED=1 uv pip install --editable .
   cd -
   ```
   
   > **Note:** Make sure to enable vLLM V1.

## 📊 Usage

### Running Experiments

**Self-Consistency Baseline:**
```bash
python deepconf-baseline.py --qid 0 --rid 0
```

**DeepConf Online Method:**
```bash
python deepconf-online.py --qid 0 --rid 0
```

### Parameters

- `--qid`: Question ID (range: 0-29)
- `--rid`: Run ID (any integer for experiment tracking)

### Configuration

Modify these settings in the Python files for different experimental configurations:

```python
MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_TOKENS = 64000
DATASET_FILE = "brumo_2025.jsonl"
WARMUP_TRACES = 16
TOTAL_BUDGET = 256
CONFIDENCE_PERCENTILE = 90  # See configuration guide below
```

### Confidence Percentile Configuration

- **DeepConf-Low** (`CONFIDENCE_PERCENTILE = 90`): Keeps top 10% of responses
  - Prioritizes token efficiency
  - Selects only the highest confidence responses
  
- **DeepConf-High** (`CONFIDENCE_PERCENTILE = 10`): Keeps top 90% of responses
  - More comprehensive response selection
  - Higher computational cost but potentially better coverage

> **Important:** DeepConf-Low doesn't guarantee better performance than the baseline, but either DeepConf-Low or DeepConf-High should outperform the self-consistency baseline depending on your dataset and use case.

### Running Complete Experiments

For statistically valid results, ensure equal numbers of runs across all questions:

```bash
# Example: Run all questions with 8 runs each
for qid in {0..29}; do
    for rid in {0..7}; do
        python deepconf-baseline.py --qid $qid --rid $rid
        python deepconf-online.py --qid $qid --rid $rid
    done
done
```

### Analysis

After completing all experiments:
```bash
python analysis.py
```

## 📈 Expected Results

Based on our experiments with DeepSeek-8B on BRUMO25 dataset (total budget: 256, 1 runs per question, evaluated on 1H200 GPU, keep top10% most confidence traces):

You should expect accuracy improvement and token/time savings.

```
==================================================
DEEPCONF RESULTS ANALYSIS
==================================================
Total experiments: 480
Self-Consistency method: 240 experiments
DeepConf method: 240 experiments
Questions covered: 30

----------------------------------------
METHOD COMPARISON
----------------------------------------
Self-Consistency:
  Accuracy: 90.8% (218/240)
  Avg tokens: 5929181
  Avg time: 1050.1s
DeepConf:
  Accuracy: 93.3% (224/240)
  Avg tokens: 2793706
  Avg time: 725.0s

----------------------------------------
DELTA ANALYSIS (DeepConf vs Baseline)
----------------------------------------

Accuracy delta: +2.5%
Token delta: -3135475 tokens (-52.9%)
Time delta: -325.2s (-31.0%)
```

## License
DeepConf is MIT licensed, as found in the LICENSE file.