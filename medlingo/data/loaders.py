"""
medlingo/data/loaders.py

Dataset loading utilities for all four MedLingo training stages.

All loaders follow a consistent contract:
  - Accept a `cache_dir` argument to persist downloaded data.
  - Return a HuggingFace `datasets.Dataset` compatible object.
  - Fall back gracefully to a synthetic stub when the real dataset
    is unavailable (e.g., gated HF Hub repos, missing HF_TOKEN).

Datasets per stage (from §IV):
  Stage 1 : LLaVA-Med           (~600k image-caption pairs)
  Stage 2 : PathVQA, RadVQA,    (~300k multimodal VQA pairs)
            SLAKE, PMC-VQA
  Stage 3 : MedQA, PubMedQA     (~50k query-specialty pairs)
  Stage 4 : MIMIC-IV-VQA,       (preference triplets)
            Quilt-1M, MedMNIST-Brain,
            MedQA-USMLE, PAD-UFES-20
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# HuggingFace Hub identifiers for each dataset
_HF_IDS = {
    "llava_med":        "microsoft/LLaVA-Med",
    "pathvqa":          "flaviagiammarino/path-vqa",
    "radvqa":           "Aniket-kumar-7/rad-vqa",
    "slake":            "mdwiratathya/SLAKE-vqa",
    "pmc_vqa":          "xmcmic/PMC-VQA",
    "medqa":            "GBaker/MedQA-USMLE-4-options",
    "pubmedqa":         "qiaojin/PubMedQA",
    "mimic_iv_vqa":     "mdwiratathya/MIMIC-IV-VQA",
    "quilt_1m":         "wisdomik/Quilt-LLaVA-Instruct-107K",
    "medmnist_brain":   "albertvillanova/medmnist-v2",
    "pad_ufes_20":      "brunobraga/pad-ufes-20-dataset",
}

_DOMAIN_DATASET_MAP = {
    "pathology":        "pathvqa",
    "radiology":        "radvqa",
    "dermatology":      "pad_ufes_20",
    "cardiology":       "mimic_iv_vqa",
    "neurology":        "medmnist_brain",
    "general_medicine": "pmc_vqa",
}


# ---------------------------------------------------------------------------
# Stage 1 loader
# ---------------------------------------------------------------------------

def load_llava_med(
    cache_dir: str,
    max_samples: int = 600_000,
    split: str = "train",
) -> "datasets.Dataset":
    """
    Load LLaVA-Med image-caption pairs for Stage 1 projection bridge training.

    Returns columns: pixel_values, input_ids, labels, attention_mask.
    Falls back to a synthetic stub if the dataset cannot be fetched.
    """
    return _load_with_fallback(
        hf_id=_HF_IDS["llava_med"],
        stub_fn=_stub_llava_med,
        max_samples=max_samples,
        cache_dir=cache_dir,
        split=split,
        stage=1,
    )


# ---------------------------------------------------------------------------
# Stage 2 loaders
# ---------------------------------------------------------------------------

def load_specialist_datasets(
    domains: List[str],
    cache_dir: str,
    max_samples_per_domain: int = 50_000,
) -> Dict[str, "datasets.Dataset"]:
    """
    Load domain-specific VQA datasets for Stage 2 specialist training.

    Returns a dict mapping domain name → HuggingFace Dataset with columns:
    question, answer, image_path (or pixel_values if images pre-loaded).
    """
    result = {}
    for domain in domains:
        ds_key = _DOMAIN_DATASET_MAP.get(domain)
        if ds_key is None:
            logger.warning("No dataset mapping for domain '%s' — using stub.", domain)
            result[domain] = _stub_vqa(domain, max_samples_per_domain)
            continue

        result[domain] = _load_with_fallback(
            hf_id=_HF_IDS[ds_key],
            stub_fn=lambda n=domain, m=max_samples_per_domain: _stub_vqa(n, m),
            max_samples=max_samples_per_domain,
            cache_dir=cache_dir,
            stage=2,
        )
        logger.info("Loaded %s dataset: %d samples", domain, len(result[domain]))

    return result


# ---------------------------------------------------------------------------
# Stage 3 loader
# ---------------------------------------------------------------------------

def load_triage_dataset(
    cache_dir: str,
    max_samples: int = 50_000,
) -> "datasets.Dataset":
    """
    Load MedQA + PubMedQA for router triage training.

    Returns columns: query, top_domains (list[str]), domain_scores (dict).
    """
    return _load_with_fallback(
        hf_id=_HF_IDS["medqa"],
        stub_fn=_stub_triage,
        max_samples=max_samples,
        cache_dir=cache_dir,
        stage=3,
    )


# ---------------------------------------------------------------------------
# Stage 4 loader
# ---------------------------------------------------------------------------

def load_dpo_dataset(
    cache_dir: str,
    max_samples: int = 20_000,
) -> "datasets.Dataset":
    """
    Load preference-annotated pairs for DPO adjudicator alignment.

    Returns columns: prompt, chosen (grounded report), rejected (hallucinated).
    """
    return _load_with_fallback(
        hf_id=_HF_IDS["mimic_iv_vqa"],
        stub_fn=_stub_dpo,
        max_samples=max_samples,
        cache_dir=cache_dir,
        stage=4,
    )


# ---------------------------------------------------------------------------
# Generic loader with stub fallback
# ---------------------------------------------------------------------------

def _load_with_fallback(
    hf_id: str,
    stub_fn,
    max_samples: int,
    cache_dir: str,
    stage: int,
    split: str = "train",
) -> "datasets.Dataset":
    """
    Attempt to load a dataset from HuggingFace Hub.
    Falls back to a deterministic synthetic stub on any failure.
    """
    try:
        import datasets as hf_datasets
        from medlingo.utils.model_registry import get_registry
        registry = get_registry()

        logger.info("Loading dataset: %s (max_samples=%d)", hf_id, max_samples)
        ds = hf_datasets.load_dataset(
            hf_id,
            split=split,
            cache_dir=cache_dir,
            token=registry.hf_token,
            trust_remote_code=True,
        )

        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        logger.info("Dataset loaded: %s — %d samples", hf_id, len(ds))
        return ds

    except Exception as exc:
        logger.warning(
            "Stage %d dataset '%s' unavailable (%s). "
            "Falling back to synthetic stub. "
            "Provide a valid HUGGINGFACE_TOKEN in .env for real data.",
            stage, hf_id, exc,
        )
        return stub_fn()


# ---------------------------------------------------------------------------
# Synthetic stubs — structurally identical to real datasets
# ---------------------------------------------------------------------------

def _stub_llava_med(n: int = 1_000) -> "datasets.Dataset":
    """
    Synthetic Stage 1 stub.
    Each sample mirrors the LLaVA-Med schema: image + caption.
    """
    import datasets as hf_datasets
    import torch

    rng = random.Random(42)
    templates = [
        ("chest_xray", "Chest radiograph demonstrating bilateral infiltrates consistent with pneumonia."),
        ("brain_mri",  "Axial T2-weighted MRI showing hyperintense lesion in left temporal lobe."),
        ("skin_lesion","Dermoscopic image of an asymmetric pigmented lesion with irregular border."),
        ("pathology",  "H&E stained section revealing poorly differentiated adenocarcinoma cells."),
        ("ecg_strip",  "12-lead ECG demonstrating ST-segment elevation in leads V1-V4."),
    ]

    samples = {
        "image_path":    [],
        "caption":       [],
        "input_ids":     [],
        "labels":        [],
        "attention_mask":[],
        "pixel_values":  [],
    }

    for i in range(n):
        img_type, caption = templates[i % len(templates)]
        samples["image_path"].append(f"/data/llava_med/{img_type}_{i:06d}.jpg")
        samples["caption"].append(caption)
        seq_len = rng.randint(32, 128)
        ids = [rng.randint(1, 32000) for _ in range(seq_len)]
        samples["input_ids"].append(ids)
        samples["labels"].append(ids)
        samples["attention_mask"].append([1] * seq_len)
        # Placeholder pixel values (3, 224, 224) normalized
        samples["pixel_values"].append([[[[0.5] * 224] * 224] * 3])

    return hf_datasets.Dataset.from_dict(samples)


def _stub_vqa(domain: str, n: int = 500) -> "datasets.Dataset":
    """
    Synthetic Stage 2 stub for a given domain.
    Mirrors the VQA schema: question, answer, image_path.
    """
    import datasets as hf_datasets

    domain_qa = {
        "pathology": [
            ("What cell type is predominant in this biopsy?",
             "The biopsy shows predominantly atypical squamous cells with hyperchromatic nuclei."),
            ("Is there evidence of malignancy?",
             "Yes, the section demonstrates invasive ductal carcinoma with stromal desmoplasia."),
        ],
        "radiology": [
            ("Is there evidence of consolidation in this chest X-ray?",
             "There is right lower lobe consolidation consistent with lobar pneumonia."),
            ("What does this CT scan of the abdomen reveal?",
             "CT demonstrates a 3.2 cm hypodense lesion in hepatic segment VI, suspicious for metastasis."),
        ],
        "dermatology": [
            ("What type of skin lesion is shown?",
             "The image shows a 12mm asymmetric melanocytic lesion with variegated pigmentation."),
            ("Does this lesion require biopsy?",
             "Yes, the irregular border and multiple colors (ABCDE criteria) warrant excisional biopsy."),
        ],
        "cardiology": [
            ("What abnormality is present on this ECG?",
             "There is ST-elevation in V1-V4 consistent with anterior STEMI."),
            ("Describe the echocardiographic finding.",
             "The echo demonstrates reduced LV ejection fraction of approximately 35% with apical hypokinesis."),
        ],
        "neurology": [
            ("What does this brain MRI show?",
             "T2/FLAIR sequence demonstrates a 2.1 cm hyperintense lesion in the right frontal white matter."),
            ("Is there evidence of infarct?",
             "DWI sequence confirms acute ischemic infarction in the left MCA territory."),
        ],
        "general_medicine": [
            ("What is the likely diagnosis based on these findings?",
             "Clinical and radiological findings are consistent with community-acquired pneumonia."),
            ("What is the differential diagnosis?",
             "Primary differential includes pulmonary edema, atypical pneumonia, and malignancy."),
        ],
    }

    qa_pairs = domain_qa.get(domain, domain_qa["general_medicine"])
    rng = random.Random(42)

    samples = {"question": [], "answer": [], "image_path": [], "domain": []}
    for i in range(n):
        q, a = qa_pairs[i % len(qa_pairs)]
        samples["question"].append(q)
        samples["answer"].append(a)
        samples["image_path"].append(f"/data/{domain}/sample_{i:06d}.jpg")
        samples["domain"].append(domain)

    import datasets as hf_datasets
    return hf_datasets.Dataset.from_dict(samples)


def _stub_triage(n: int = 1_000) -> "datasets.Dataset":
    """
    Synthetic Stage 3 stub for router triage training.
    Provides query, top_domains, domain_scores columns.
    """
    import datasets as hf_datasets

    samples_data = [
        ("Patient presents with haemoptysis and a right hilar mass on chest X-ray.",
         ["radiology", "pathology"],
         {"pathology": 0.35, "radiology": 0.40, "dermatology": 0.02,
          "cardiology": 0.08, "neurology": 0.05, "general_medicine": 0.10}),
        ("Dermoscopy reveals an asymmetric lesion with atypical vascular pattern.",
         ["dermatology", "pathology"],
         {"pathology": 0.30, "radiology": 0.05, "dermatology": 0.45,
          "cardiology": 0.03, "neurology": 0.02, "general_medicine": 0.15}),
        ("ECG shows prolonged QTc and U waves. Patient complains of palpitations.",
         ["cardiology", "general_medicine"],
         {"pathology": 0.03, "radiology": 0.05, "dermatology": 0.02,
          "cardiology": 0.55, "neurology": 0.05, "general_medicine": 0.30}),
        ("MRI brain shows periventricular white matter changes in a 68-year-old.",
         ["neurology", "radiology"],
         {"pathology": 0.08, "radiology": 0.30, "dermatology": 0.01,
          "cardiology": 0.06, "neurology": 0.50, "general_medicine": 0.05}),
        ("Histology slide shows Reed-Sternberg cells in a lymph node biopsy.",
         ["pathology", "general_medicine"],
         {"pathology": 0.60, "radiology": 0.10, "dermatology": 0.02,
          "cardiology": 0.03, "neurology": 0.05, "general_medicine": 0.20}),
    ]

    samples = {"query": [], "top_domains": [], "domain_scores": []}
    for i in range(n):
        q, domains, scores = samples_data[i % len(samples_data)]
        samples["query"].append(q)
        samples["top_domains"].append(domains)
        samples["domain_scores"].append(scores)

    return hf_datasets.Dataset.from_dict(samples)


def _stub_dpo(n: int = 500) -> "datasets.Dataset":
    """
    Synthetic Stage 4 stub with preference triplets.
    Mirrors (prompt, chosen, rejected) for DPO training.
    """
    import datasets as hf_datasets

    triplets = [
        (
            "Analyze this chest X-ray for signs of pulmonary pathology.",
            # Chosen: grounded, references specific visual features
            "The posteroanterior chest radiograph demonstrates a 4.2 cm opacity in the right "
            "lower lobe with air bronchograms, consistent with lobar pneumonia. The left lung "
            "field is clear. No pleural effusion is identified. These findings are directly "
            "observable in the provided image.",
            # Rejected: hallucinated, no visual grounding
            "The patient likely has bilateral pneumonia with associated pleural effusions and "
            "mediastinal shift. This is a severe case requiring immediate ICU admission.",
        ),
        (
            "Describe the pathological findings in this H&E stained tissue section.",
            "The H&E section demonstrates sheets of malignant epithelial cells with high "
            "nuclear-to-cytoplasmic ratio, prominent nucleoli, and frequent mitotic figures "
            "visible at 40x magnification. Stromal invasion is present at the image periphery.",
            "This tissue shows signs of chronic inflammation and fibrosis typical of "
            "autoimmune hepatitis, with interface hepatitis and plasma cell infiltrates.",
        ),
        (
            "What abnormality does this brain MRI reveal?",
            "The axial T2-weighted sequence reveals a well-circumscribed 2.8 cm hyperintense "
            "lesion in the right parietal lobe with surrounding vasogenic oedema, suggesting "
            "a high-grade glioma or metastatic deposit. Mass effect causes 4mm midline shift.",
            "The brain MRI is entirely normal with no focal lesions, normal grey-white "
            "differentiation, and no evidence of acute pathology.",
        ),
        (
            "Interpret the dermatological findings in this clinical photograph.",
            "The clinical photograph shows a 15mm asymmetric pigmented lesion on the upper back "
            "with irregular scalloped borders, variegated colour (brown, black, and focal grey), "
            "meeting three of the ABCDE criteria for melanoma. Urgent dermatological review advised.",
            "This appears to be a benign seborrhoeic keratosis with typical stuck-on appearance "
            "and regular border. No further investigation is required.",
        ),
        (
            "What does this ECG demonstrate?",
            "The 12-lead ECG shows 2mm ST-segment elevation in leads V1 through V4 with "
            "reciprocal ST-depression in the inferior leads (II, III, aVF), consistent with "
            "acute anterior STEMI. Immediate cardiology consult and primary PCI indicated.",
            "The ECG shows a normal sinus rhythm with no significant ST-segment changes. "
            "The patient can be safely discharged with outpatient follow-up.",
        ),
    ]

    samples = {"prompt": [], "chosen": [], "rejected": []}
    for i in range(n):
        p, c, r = triplets[i % len(triplets)]
        samples["prompt"].append(p)
        samples["chosen"].append(c)
        samples["rejected"].append(r)

    return hf_datasets.Dataset.from_dict(samples)
