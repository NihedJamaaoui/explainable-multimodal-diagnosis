#!/usr/bin/env python3
# =====================================================================
# 03_encoder_jpg.py
# À lancer sur GPU (via job SLURM). Encode les images JPG déjà
# téléchargées en patches (BiomedCLIP), déduplique par étude, et
# aligne l'EHR. Produit les fichiers _clean prêts pour l'entraînement.
# =====================================================================

import os
import numpy as np, pandas as pd, torch
from PIL import Image
from open_clip import create_model_from_pretrained

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
RAW  = os.path.join(BASE, 'data/raw')
PROC = os.path.join(BASE, 'data/processed')
os.makedirs(PROC, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Encodage sur :', device, flush=True)

# ---------- 1) charger l'index des images JPG + dédupliquer par étude ----------
idx = pd.read_csv(os.path.join(RAW, 'image_index_jpg.csv'))
idx = idx.drop_duplicates(subset='study_id', keep='first').reset_index(drop=True)
print('Images à encoder (après dédup étude) :', len(idx), flush=True)

# ---------- 2) charger BiomedCLIP (depuis le cache, hors ligne) ----------
VISION = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
model, preprocess = create_model_from_pretrained(VISION)
model = model.to(device).eval()

def encoder(img):
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.visual.trunk.forward_features(x)   # [1,197,768]
    return tokens[:, 1:, :].squeeze(0).cpu().numpy()      # [196,768] (sans le token CLS)

# ---------- 3) encoder chaque image ----------
patches, infos = [], []
for n, r in idx.iterrows():
    try:
        img = Image.open(r['jpg_path']).convert('RGB')
        patches.append(encoder(img))
        infos.append({'subject_id': r['subject_id'], 'study_id': r['study_id'], 'dicom_id': r['dicom_id']})
    except Exception as e:
        pass
    if (n + 1) % 200 == 0:
        np.save(os.path.join(PROC, 'image_patches_clean.npy'), np.stack(patches))
        pd.DataFrame(infos).to_csv(os.path.join(PROC, 'image_index_clean.csv'), index=False)
        print(f'  ... {n+1} traitées, {len(patches)} encodées (sauvegardé)', flush=True)

np.save(os.path.join(PROC, 'image_patches_clean.npy'), np.stack(patches))
pd.DataFrame(infos).to_csv(os.path.join(PROC, 'image_index_clean.csv'), index=False)
print('Patches encodés :', len(patches), '| forme :', patches[0].shape, flush=True)

# ---------- 4) aligner l'EHR (garder les patients qui ont une image) ----------
index_clean = pd.read_csv(os.path.join(PROC, 'image_index_clean.csv'))
patients_img = set(index_clean['subject_id'].unique())
ehr = pd.read_csv(os.path.join(RAW, 'cohort_ehr_text.csv'))
ehr_clean = ehr[ehr['subject_id'].isin(patients_img)].reset_index(drop=True)
ehr_clean.to_csv(os.path.join(PROC, 'cohort_ehr_text_clean.csv'), index=False)
pd.DataFrame({'subject_id': sorted(patients_img)}).to_csv(os.path.join(PROC, 'cohort_final.csv'), index=False)

print('=== TERMINÉ ===', flush=True)
print('Patients finaux :', len(patients_img), '| images :', len(index_clean), flush=True)
print('Fichiers prêts dans data/processed/ : image_patches_clean.npy, image_index_clean.csv, cohort_ehr_text_clean.csv', flush=True)
