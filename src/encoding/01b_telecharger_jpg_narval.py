#!/usr/bin/env python3
# =====================================================================
# 01b_telecharger_jpg_narval.py
# À lancer sur le NŒUD DE CONNEXION de Narval (internet) — SANS GPU.
# Télécharge les images MIMIC-CXR-JPG (LÉGÈRES) pour une grande cohorte.
# Avantages : ~10x plus léger que le DICOM, téléchargement PARALLÈLE,
# timeout robuste (ne bloque plus), reprise automatique.
# Réutilise l'EHR déjà téléchargé (cohort_ehr_text.csv, 8000 patients).
# =====================================================================

import os, subprocess, csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
RAW  = os.path.join(BASE, 'data/raw')
JPG  = os.path.join(BASE, 'data/jpg')          # images JPG téléchargées
os.makedirs(JPG, exist_ok=True)

N_IMAGES   = 20000        # objectif (monte si tu veux plus)
N_PARALLEL = 8            # nombre de téléchargements simultanés
VUES_FRONTALES = ['PA', 'AP']

# MIMIC-CXR-JPG : mêmes patients, images légères
CXR_JPG = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'

def lire_identifiants():
    with open(os.path.join(BASE, '.physionet')) as f:
        m = f.read().split()
    return m[m.index('login')+1], m[m.index('password')+1]
USER, PWD = lire_identifiants()

def wget(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    r = subprocess.run(['wget', '-q', '--timeout=30', '--tries=2',
                        '--user', USER, '--password', PWD, '-O', dest, url])
    ok = r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0
    if not ok and os.path.exists(dest):
        os.remove(dest)
    return ok

# ---------- cohorte = patients de l'EHR déjà téléchargé ----------
cohorte = set(pd.read_csv(os.path.join(RAW, 'cohort_ehr_text.csv'))['subject_id'])
print('Cohorte (depuis EHR déjà téléchargé) :', len(cohorte), flush=True)

# ---------- métadonnées : quelle image est frontale ? ----------
# Le fichier mimic-cxr-2.0.0-metadata.csv.gz donne la vue (ViewPosition) par image.
print('Téléchargement des métadonnées (vues)...', flush=True)
wget(CXR_JPG + 'mimic-cxr-2.0.0-metadata.csv.gz', os.path.join(RAW, 'cxr-metadata.csv.gz'))
meta = pd.read_csv(os.path.join(RAW, 'cxr-metadata.csv.gz'))
meta = meta[meta['subject_id'].isin(cohorte) & meta['ViewPosition'].isin(VUES_FRONTALES)]
meta = meta.head(N_IMAGES)
print('Images frontales à télécharger :', len(meta), flush=True)

# ---------- construire l'URL JPG de chaque image ----------
# chemin JPG : files/pXX/pXXXXXXXX/sYYYYYYYY/<dicom_id>.jpg
def chemin_jpg(subj, study, dicom):
    p = f'p{str(subj)[:2]}/p{subj}/s{study}/{dicom}.jpg'
    return CXR_JPG + 'files/' + p, os.path.join(JPG, f'{dicom}.jpg')

taches = []
for _, r in meta.iterrows():
    url, dest = chemin_jpg(r['subject_id'], r['study_id'], r['dicom_id'])
    taches.append((r['subject_id'], r['study_id'], r['dicom_id'], url, dest, r['ViewPosition']))

# ---------- téléchargement PARALLÈLE ----------
index_path = os.path.join(RAW, 'image_index_jpg.csv')
faits = set()
if os.path.exists(index_path):
    faits = set(pd.read_csv(index_path)['dicom_id'])
    print('Déjà téléchargées :', len(faits), flush=True)

def traiter(t):
    subj, study, dicom, url, dest, vue = t
    if dicom in faits:
        return None
    if wget(url, dest):
        return {'subject_id': subj, 'study_id': study, 'dicom_id': dicom, 'jpg_path': dest, 'view': vue}
    return None

ok = len(faits); n = 0
f = open(index_path, 'a', newline='')
w = csv.DictWriter(f, fieldnames=['subject_id','study_id','dicom_id','jpg_path','view'])
if os.path.getsize(index_path) == 0:
    w.writeheader()

with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
    futures = [ex.submit(traiter, t) for t in taches]
    for fut in as_completed(futures):
        res = fut.result(); n += 1
        if res:
            w.writerow(res); ok += 1
        if n % 200 == 0:
            f.flush()
            print(f'  ... {n} traitées, {ok} images JPG gardées', flush=True)

f.close()
print('=== TERMINÉ ===', flush=True)
print('Images JPG téléchargées :', ok, flush=True)
print('Index :', index_path, flush=True)
print('Prochaine étape : encodage BiomedCLIP sur GPU (script 03).', flush=True)
