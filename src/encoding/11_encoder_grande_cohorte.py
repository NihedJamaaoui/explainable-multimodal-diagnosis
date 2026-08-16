#!/usr/bin/env python3
# =====================================================================
# 11_encoder_grande_cohorte.py
# ENCODAGE GPU GRANDE ECHELLE (Narval, job SLURM).
#
# Optimisations pour le gros volume :
#   - float16  -> divise la taille par 2 (vs float32)
#   - memmap   -> ecrit directement sur disque, pas tout en RAM
#   - batches  -> exploite le A100 (beaucoup plus rapide)
#   - reprise  -> si le job est coupe, il repart ou il en etait
#
# Produit : data/processed/image_patches_f16.npy  (memmap float16)
#           data/processed/image_index_clean.csv
# =====================================================================

import os
import numpy as np, pandas as pd, torch
from PIL import Image

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
RAW  = os.path.join(BASE, 'data/raw')
PROC = os.path.join(BASE, 'data/processed')
os.makedirs(PROC, exist_ok=True)

BATCH = 32          # images par lot sur le GPU
N_PATCHES = 196
DIM = 768

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('=== ENCODAGE GRANDE COHORTE ===', flush=True)
print('Device :', device, flush=True)

# ---------------- index des images ----------------
idx = pd.read_csv(os.path.join(RAW, 'image_index_jpg.csv'))
# une seule image par etude (securite)
idx = idx.drop_duplicates(subset=['study_id'], keep='first').reset_index(drop=True)
# ne garder que les fichiers reellement presents
idx['existe'] = idx['jpg_path'].apply(os.path.exists)
idx = idx[idx['existe']].drop(columns=['existe']).reset_index(drop=True)
N = len(idx)
print('Images a encoder :', N, '| patients :', idx['subject_id'].nunique(), flush=True)

# ---------------- modele BiomedCLIP ----------------
from open_clip import create_model_from_pretrained
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model = model.visual.trunk.eval().to(device)
print('BiomedCLIP charge.', flush=True)

# ---------------- sortie memmap float16 ----------------
out_path = os.path.join(PROC, 'image_patches_f16.npy')
etat_path = os.path.join(PROC, 'encodage_etat.txt')

deja = 0
if os.path.exists(out_path) and os.path.exists(etat_path):
    with open(etat_path) as f:
        deja = int(f.read().strip() or 0)
    patches = np.lib.format.open_memmap(out_path, mode='r+')
    print('Reprise : deja encodees =', deja, flush=True)
else:
    patches = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float16,
                                        shape=(N, N_PATCHES, DIM))
    print('Fichier memmap cree :', out_path,
          f'({N * N_PATCHES * DIM * 2 / 1e9:.1f} Go)', flush=True)

def charger(chemin):
    try:
        img = Image.open(chemin).convert('RGB')
        return preprocess(img)
    except Exception:
        return None

# ---------------- boucle d encodage par lots ----------------
i = deja
while i < N:
    lot = idx.iloc[i:i + BATCH]
    tenseurs, positions = [], []
    for k, (_, r) in enumerate(lot.iterrows()):
        t = charger(r['jpg_path'])
        if t is not None:
            tenseurs.append(t); positions.append(i + k)
    if tenseurs:
        x = torch.stack(tenseurs).to(device)
        with torch.no_grad():
            feats = model.forward_features(x)      # (B, 1+196, 768)
            feats = feats[:, 1:, :]                # on enleve le token CLS
        arr = feats.cpu().numpy().astype(np.float16)
        for j, pos in enumerate(positions):
            patches[pos] = arr[j]
    i += BATCH
    if (i // BATCH) % 20 == 0:
        patches.flush()
        with open(etat_path, 'w') as f:
            f.write(str(min(i, N)))
        print(f'  ... {min(i, N)}/{N} encodees', flush=True)

patches.flush()
with open(etat_path, 'w') as f:
    f.write(str(N))

# ---------------- index final ----------------
idx.to_csv(os.path.join(PROC, 'image_index_clean.csv'), index=False)

print('=== TERMINE ===', flush=True)
print('Patches :', out_path, '| forme :', patches.shape, '| dtype : float16', flush=True)
print('Index   :', os.path.join(PROC, 'image_index_clean.csv'), flush=True)
print('Patients :', idx['subject_id'].nunique(), '| Images :', N, flush=True)
print("Prochaine etape : EHR enrichi (script 02) sur cette nouvelle cohorte.", flush=True)
