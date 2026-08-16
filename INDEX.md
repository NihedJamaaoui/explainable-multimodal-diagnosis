\# Script Index



A guide to the scripts, organised by pipeline stage.



\## Encoding (src/encoding/)



| Script | Purpose |

|--------|---------|

| 01\_telecharger\_narval.py | Download and link MIMIC-IV / MIMIC-CXR cohort |

| 01b\_telecharger\_jpg\_narval.py | Download the chest X-ray JPG images |

| 02\_ehr\_enrichi\_labels\_narval.py | Build enriched EHR triplets and labels |

| 03\_encoder\_jpg.py | Encode images with BiomedCLIP-ViT (patches) |

| 10\_selection\_grande\_cohorte.py | Select the large cohort |

| 11\_encoder\_grande\_cohorte.py | Encode the full cohort |



\## Training and grounding (src/training/)



| Script | Purpose |

|--------|---------|

| 12\_entrainement\_complet.py | Train the fusion model with the composite loss |

| 31\_preparer\_imagenome.py | Prepare Chest ImaGenome anatomical boxes |

| 31b\_corriger\_dimensions.py | Fix the ImaGenome coordinate-frame issue |

| 32\_lground.py | Anatomical grounding (L\_ground) — core contribution |



\## Experiments (experiments/)



| Script | Purpose |

|--------|---------|

| 13\_recherche\_Lconsist\_variantes.py | Compare five L\_consist formulations |

| 15\_comparaison\_MSE\_KL\_multitache.py | MSE vs KL across tasks |

| 25\_multilabel\_variantes\_seeds.py | Multi-label consistency, three seeds |

| 30\_iou\_mscxr.py | Grounding validation against MS-CXR expert boxes |

| medfuse\_notre\_cohort.py | MedFuse LSTM fusion on our cohort |

| comparer\_fusions.py | Early / Late / DrFuse fusion baselines |



\## Report generation (src/generation/)



| Script | Purpose |

|--------|---------|

| 40\_base\_connaissance.py | Build the RAG knowledge base (804 passages) |

| 44\_rapports\_cohorte.py | Generate reports over the cohort |

| 49\_rougeL\_rapports\_reels.py | Generate real reports and compute ROUGE-L |



\## Evaluation (src/evaluation/)



| Script | Purpose |

|--------|---------|

| 20\_evaluation\_fairness.py | Fairness across sex and age subgroups |

| 23\_detection\_multimaladie.py | Multi-disease detection |

| 24\_multilabel\_correct.py | Multi-label classification metrics (AUROC, AUPRC, bootstrap) |

