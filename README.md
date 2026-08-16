\# Grounded and Consistent Multimodal Learning for Explainable Chest Disease Diagnosis



\*Fusing Chest Radiographs (MIMIC-CXR) and Electronic Health Records (MIMIC-IV)\*



Code for a Master's research thesis (UQAR / Mitacs Globalink) on an

explainable multimodal framework that fuses chest radiographs and electronic

health records, with two contributions: \*\*anatomical grounding\*\* (L\_ground)

and \*\*cross-modal consistency\*\* (L\_consist).



\## Overview



A frozen BiomedCLIP-ViT image encoder and a frozen Bio\_ClinicalBERT text

encoder are fused through cross-attention, trained with a composite objective:



&#x20;   L\_total = L\_task + lambda1 \* L\_ground + lambda2 \* L\_consist



A retrieval-augmented stage (RAG + Llama-3.1) turns the grounded findings

into a written diagnostic report.



\## Key results (test set, 2,779 images)



| Metric | Value |

|--------|-------|

| Mean AUROC | 0.808 |

| Mean AUPRC | 0.401 |

| L\_ground: attention mass in region | 0.48 -> 0.97 (3 seeds) |

| L\_consist: cross-modal disagreement | -59% |

| Report generation: finding recall | 0.71 |



\### Direct comparison of fusion strategies (same cohort, same inputs)



| Fusion | AUROC | AUPRC |

|--------|-------|-------|

| MedFuse (LSTM) | 0.754 | 0.334 |

| Early (concat) | 0.757 | 0.340 |

| Late (average) | 0.757 | 0.347 |

| DrFuse (latent) | 0.757 | 0.344 |

| Cross-attention (ours) | 0.808 | 0.401 |



\## Repository structure



&#x20;   src/encoding/     data encoding (BiomedCLIP patches, Bio\_ClinicalBERT triplets)

&#x20;   src/training/     composite-loss training, L\_ground, ImaGenome preparation

&#x20;   src/generation/   RAG knowledge base + report generation

&#x20;   src/evaluation/   classification, fairness, detection

&#x20;   experiments/      consistency variants, MedFuse and fusion comparisons, MS-CXR IoU



\## Data access



This work uses restricted-access datasets from PhysioNet (MIMIC-CXR,

MIMIC-IV, MS-CXR, Chest ImaGenome). The data is NOT included here; access

requires credentialed PhysioNet approval.



\## Reproducibility



The whole framework trains on a single A100 GPU (Alliance Canada / Narval).

Compute nodes have no internet, so pre-cache the models

(HF\_HUB\_OFFLINE=1, TRANSFORMERS\_OFFLINE=1).



\## License



Code under the MIT License. Datasets remain under their PhysioNet licenses.

