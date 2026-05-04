# MedLingo

**Resource-Efficient Hierarchical Multi-Agent Intelligence for Autonomous Clinical and Robotic Diagnostics**

> *Sultan et al., ICRAI 2026 — National University of Sciences and Technology (NUST), Islamabad*

---

MedLingo is a hierarchical multi-agent framework that delivers high-fidelity medical image diagnostics within a **16 GB VRAM budget** on a single consumer GPU. It orchestrates an ensemble of specialised 1B-parameter expert models under the supervision of an 8B-parameter adjudicator — achieving diagnostic accuracy comparable to much larger monolithic models while remaining deployable on edge hardware.

---

## Architecture

```
Medical Image + Clinical Query
          │
          ▼
  ┌───────────────────┐
  │  BioMedCLIP (FP16)│   ← Frozen vision encoder — 768-dim features
  └────────┬──────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
 Pathway A     Pathway B        ← Dual-path linear projection bridges
 (768→2048)   (768→4096)
     │            │
     ▼            └──────────────────────────────────┐
  ┌──────────────────────┐                           │
  │  Semantic Triage     │   ← Llama-3.2-1B NF4      │
  │  Router (top-2)      │                           │
  └──────────┬───────────┘                           │
             │ selects 2 of 6 domains                │
     ┌───────┴────────┐                              │
     │                │                              │
     ▼                ▼                              │
 Instance 1       Instance 2    ← Parallel LoRA      │
 (Specialist A)   (Specialist B)  Hot-Swap           │
     │                │                              │
     └───────┬────────┘                              │
             │ Semantic Peer Review (SPR)             │
             │ (entropy-weighted, max 2 rounds)       │
             ▼                                       │
  ┌──────────────────────┐                           │
  │  Chief Adjudicator   │ ◄─────────────────────────┘
  │  (Llama-3.1-8B NF4)  │   Pathway B visual anchor
  │  DPO-aligned         │   + full SPR transcript
  └──────────────────────┘
             │
             ▼
    Grounded Clinical Verdict
```

### Components (Table I)

| Component       | Backbone           | Quant. | Latent Dim | Role              |
|-----------------|--------------------|--------|------------|-------------------|
| Vision Encoder  | BioMedCLIP         | FP16   | 768        | Feature Extraction |
| Router          | Llama-3.2-1B       | NF4    | 2048       | Semantic Triage   |
| Specialists ×6  | Llama-3.2-1B       | NF4    | 2048       | VQA + SPR         |
| Adjudicator     | Llama-3.1-8B       | NF4    | 4096       | Adjudication      |

**Peak VRAM: 14.8 GB** (within 16 GB budget on RTX 4080/4090/A4000)

---

## Training Pipeline

MedLingo is trained in four sequential stages, each building on the previous:

| Stage | Objective                  | Dataset                        | Key Config         |
|-------|----------------------------|--------------------------------|--------------------|
| 1     | Dual-path bridge alignment | LLaVA-Med (600k pairs)         | LR=1e-3, cosine    |
| 2     | Specialist injection       | PathVQA+RadVQA+SLAKE+PMC-VQA   | LoRA r=64, α=128   |
| 3     | Router triage calibration  | MedQA + PubMedQA (50k)         | LR=1e-5, linear    |
| 4     | DPO adjudicator alignment  | MIMIC+Quilt-1M+PAD-UFES+MedQA  | β=0.1, LoRA r=128  |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/medlingo.git
cd medlingo
pip install -e ".[dev]"
```

For FlashAttention-2 (required for Stage 4 and inference):
```bash
pip install flash-attn --no-build-isolation
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set MEDLINGO_WEIGHTS_DIR and HUGGINGFACE_TOKEN
```

### 3. Accept model licences on HuggingFace Hub

The following gated models require licence acceptance before downloading:
- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

## Training

```bash
# Full 4-stage pipeline
./scripts/train.sh --stages 1 2 3 4

# Train only the radiology specialist (Stage 2, single domain)
./scripts/train.sh --stages 2 --domain radiology

# Resume from Stage 3 (skips completed stages)
./scripts/train.sh --stages 3 4 --resume
```

Or directly via Python:
```bash
python -m medlingo.training.train_pipeline --stages 1 2 3 4
```

---

## Inference

### Interactive session

```bash
./scripts/infer.sh --interactive
```

```
medlingo> image chest_xray.jpg
Image set: chest_xray.jpg

medlingo> Is there evidence of pneumothorax?

Analyzing...

═══ MedLingo Clinical Verdict ═══

No pneumothorax identified. The left lung field shows increased
opacity in the lower zone consistent with consolidation or atelectasis.
The right hemithorax is clear. Costophrenic angles are preserved.

Visual Grounding: Increased opacity visible in left lower zone;
trachea remains midline; no visible pleural edge.

Diagnostic Confidence: HIGH
═════════════════════════════════

[3812 ms | HIGH]
```

### Single query

```bash
python -m medlingo.inference.cli \
    --image scan.jpg \
    --query "What abnormalities are present?" \
    --verbose
```

### Python API

```python
from medlingo import MedLingoEngine, DiagnosticRequest

with MedLingoEngine(device="auto") as engine:
    response = engine.diagnose(DiagnosticRequest(
        query="Is there evidence of pneumothorax?",
        image_path="chest_xray.jpg",
    ))
    print(response)
    print(response.to_dict())
```

### Batch inference

```bash
python -m medlingo.inference.cli \
    --batch queries.json \
    --output results.json
```

`queries.json` format:
```json
[
  {"id": "q1", "query": "Any consolidation?", "image_path": "scan1.jpg"},
  {"id": "q2", "query": "Describe the lesion.", "image_path": "scan2.jpg"}
]
```

---

## Evaluation

```bash
# Evaluate across all six domains (50 samples each)
python scripts/evaluate.py --domains all --samples 50

# Radiology + pathology only
python scripts/evaluate.py --domains radiology pathology --samples 100
```

### Reported results (Table III)

| Model               | Rad.  | Path. | Cardio. | Derm. | Neuro. | Gen.  | BERT-F1 |
|---------------------|-------|-------|---------|-------|--------|-------|---------|
| LLaVA-Med (7B)      | 61.5% | 38.0% | 54.8%   | 60.2% | 63.8%  | 64.5% | 0.671   |
| Llama-3.1-8B-Aloe   | 71.2% | 67.8% | 70.1%   | 65.4% | 68.1%  | 68.1% | 0.666   |
| **MedLingo (Ours)** | **82.7%** | **91.9%** | **83.8%** | **78.4%** | **75.4%** | **74.7%** | **0.697** |

---

## Project Structure

```
medlingo/
├── configs/
│   ├── model_config.yaml          # Model architecture settings
│   ├── training_config.yaml       # Stage-by-stage hyperparameters (Table II)
│   └── inference_config.yaml      # Runtime inference settings
├── medlingo/
│   ├── core/
│   │   ├── vision_encoder.py      # Frozen BioMedCLIP wrapper
│   │   ├── projection_bridges.py  # PathwayA (768→2048) + PathwayB (768→4096)
│   │   ├── uncertainty.py         # Shannon entropy scoring (Eq. 3, 5)
│   │   ├── router.py              # Semantic Triage Router
│   │   ├── specialist.py          # LoRA specialist expert + SPR refinement
│   │   └── adjudicator.py         # DPO-aligned Chief Adjudicator
│   ├── pipeline/
│   │   ├── medlingo_pipeline.py   # End-to-end inference orchestrator
│   │   ├── spr.py                 # Semantic Peer Review loop
│   │   └── parallel_executor.py  # Parallel hot-swap executor
│   ├── training/
│   │   ├── stage1_alignment.py    # Bridge alignment (LLaVA-Med)
│   │   ├── stage2_specialist.py   # Domain LoRA injection
│   │   ├── stage3_router.py       # Triage router SFT
│   │   ├── stage4_dpo.py          # DPO adjudicator alignment
│   │   └── train_pipeline.py      # 4-stage orchestrator + CLI
│   ├── data/
│   │   ├── loaders.py             # Dataset loaders with HF stubs
│   │   └── preprocessing.py       # Image + text preprocessing
│   ├── inference/
│   │   ├── engine.py              # Production inference engine
│   │   └── cli.py                 # Interactive + batch CLI
│   └── utils/
│       ├── model_registry.py      # Env-based path resolution (hides weights)
│       ├── quantization.py        # NF4 BitsAndBytes helpers
│       ├── memory_manager.py      # 16 GB VRAM budget tracker
│       └── logging_utils.py       # Rich logging + stage timers
├── scripts/
│   ├── train.sh                   # GPU training launcher
│   ├── infer.sh                   # Inference launcher
│   └── evaluate.py                # Multi-domain benchmarking
├── tests/
│   ├── test_core.py               # Unit tests (CPU-only, no weights needed)
│   └── test_pipeline.py           # Integration tests (mocked LLMs)
├── .env.example                   # Environment variable template
├── .gitignore
└── pyproject.toml
```

---

## Citation

```bibtex
@inproceedings{sultan2026medlingo,
  title        = {MedLingo: Resource-Efficient Hierarchical Multi-Agent Intelligence
                  for Autonomous Clinical and Robotic Diagnostics},
  author       = {Sarmad Sultan and Muhammad Sami Ullah and Ahmad Jan and
                  Muhammad Dawood Rizwan and Muhammad Naseer Bajwa and
                  Muhammad Moazam Fraz},
  booktitle    = {Proceedings of the IEEE 7th International Conference on
                  Robotics and Automation in Industry (ICRAI)},
  year         = {2026},
  organization = {IEEE},
  institution  = {National University of Sciences and Technology (NUST),
                  Islamabad, Pakistan and University of Staffordshire,
                  Stoke-on-Trent, United Kingdom}
}
```

---

## Acknowledgements

This work received funding from the German Academic Exchange Service (DAAD) under Project ID 8979614: *Ba-Ikhtiyar Jawan: Upscaling and Digitization of Vocational Education Curriculum.*
