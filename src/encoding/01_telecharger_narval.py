#!/usr/bin/env python3
# =====================================================================
# 01_telecharger_narval.py
# À lancer sur le NŒUD DE CONNEXION de Narval (qui a internet) — SANS GPU.
# Il télécharge une grande cohorte et prépare :
#   - le texte EHR (âge, sexe, diagnostics -> triplets)
#   - les images DICOM (vues frontales), avec un index
# L'ENCODAGE des images (BiomedCLIP) se fera APRÈS, sur GPU (script 02).
# =====================================================================

import os, subprocess, sys
import pandas as pd
import pydicom

# ---------------- CONFIGURATION ----------------
BASE      = os.path.expanduser('~/scratch/MultimodalVLM')
DATA      = os.path.join(BASE, 'data')
RAW       = os.path.join(DATA, 'raw')        # tables + fichiers légers
DICOMDIR  = os.path.join(DATA, 'dicom')      # images DICOM téléchargées
os.makedirs(RAW, exist_ok=True)
os.makedirs(DICOMDIR, exist_ok=True)

N_PATIENTS = 8000       # nombre de patients visés (monte si tu veux plus)
N_IMAGES   = 30000       # nombre max d'images à télécharger
VUES_FRONTALES = ['PA', 'AP']

IV  = 'https://physionet.org/files/mimiciv/3.1/'
CXR = 'https://physionet.org/files/mimic-cxr/2.1.0/'

# ---------------- IDENTIFIANTS PHYSIONET ----------------
# Lit ~/scratch/MultimodalVLM/.physionet  (format : machine physionet.org login X password Y)
def lire_identifiants():
    chemin = os.path.join(BASE, '.physionet')
    with open(chemin) as f:
        mots = f.read().split()
    return mots[mots.index('login')+1], mots[mots.index('password')+1]

USER, PWD = lire_identifiants()

def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    r = subprocess.run(['wget', '-q', '--timeout=30', '--tries=2',
                        '--user', USER, '--password', PWD, '-O', dest, url])
    ok = r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0
    if not ok and os.path.exists(dest):
        os.remove(dest)   # supprime le fichier vide/incomplet
    return ok

# =====================================================================
# 1) EHR : télécharger leBs tables MIMIC-IV et construire les triplets
# =====================================================================
print('=== 1) Téléchargement des tables MIMIC-IV ===', flush=True)
for f in ['hosp/patients.csv.gz', 'hosp/diagnoses_icd.csv.gz', 'hosp/d_icd_diagnoses.csv.gz']:
    dest = os.path.join(RAW, os.path.basename(f))
    print('  ', f, '...', flush=True)
    download(IV + f, dest)

patients   = pd.read_csv(os.path.join(RAW, 'patients.csv.gz'))
diagnoses  = pd.read_csv(os.path.join(RAW, 'diagnoses_icd.csv.gz'))
d_icd      = pd.read_csv(os.path.join(RAW, 'd_icd_diagnoses.csv.gz'))

# noms des maladies
diag = diagnoses.merge(d_icd, on=['icd_code', 'icd_version'], how='left')
diag = diag.dropna(subset=['long_title'])

# patients valides : sexe + âge + au moins un diagnostic
pats_ok = patients[(patients['gender'].isin(['M', 'F'])) & (patients['anchor_age'].notna())].copy()
pats_ok = pats_ok[pats_ok['subject_id'].isin(diag['subject_id'].unique())]
cohorte = pats_ok['subject_id'].head(N_PATIENTS).tolist()
print('Patients EHR retenus :', len(cohorte), flush=True)

# construire les triplets par patient
sexe_map = {'M': 'male', 'F': 'female'}
lignes = []
for _, r in pats_ok[pats_ok['subject_id'].isin(cohorte)].iterrows():
    subj = r['subject_id']
    parts = [f"patient gender {sexe_map[r['gender']]} .",
             f"patient age {int(r['anchor_age'])} ."]
    for t in diag[diag['subject_id'] == subj]['long_title'].unique():
        parts.append(f"patient has_diagnosis {t} .")
    lignes.append({'subject_id': subj, 'texte': ' '.join(parts)})
ehr_text = pd.DataFrame(lignes)
ehr_text.to_csv(os.path.join(RAW, 'cohort_ehr_text.csv'), index=False)
pd.DataFrame({'subject_id': cohorte}).to_csv(os.path.join(RAW, 'cohort_final.csv'), index=False)
print('EHR sauvegardé : cohort_ehr_text.csv (', len(ehr_text), 'patients )', flush=True)

# =====================================================================
# 2) IMAGES : télécharger les DICOM (vues frontales) de la cohorte
# =====================================================================
print('=== 2) Liste des images MIMIC-CXR ===', flush=True)
download(CXR + 'cxr-record-list.csv.gz', os.path.join(RAW, 'cxr-record-list.csv.gz'))
records = pd.read_csv(os.path.join(RAW, 'cxr-record-list.csv.gz'))
records = records[records['subject_id'].isin(cohorte)]     # seulement notre cohorte
print('Images candidates (cohorte) :', len(records), flush=True)

index_path = os.path.join(RAW, 'image_index.csv')
infos = []
if os.path.exists(index_path):
    infos = pd.read_csv(index_path).to_dict('records')
    deja = set(x['dicom_id'] for x in infos)
else:
    deja = set()

ok = len(infos); vus = 0
for _, row in records.head(N_IMAGES).iterrows():
    if row['dicom_id'] in deja:
        continue
    vus += 1
    dest = os.path.join(DICOMDIR, f"img_{row['dicom_id'][:14]}.dcm")
    if not download(CXR + row['path'], dest):
        continue
    try:
        ds  = pydicom.dcmread(dest, stop_before_pixels=True)   # header seulement (rapide)
        vue = ds.get('ViewPosition', '?')
        if vue not in VUES_FRONTALES:      # on garde la face, on écarte le profil
            os.remove(dest)                # on supprime tout de suite (économie d'espace)
            continue
        infos.append({'subject_id': row['subject_id'], 'study_id': row['study_id'],
                      'dicom_id': row['dicom_id'], 'dicom_path': dest, 'view': vue})
        ok += 1
    except Exception as e:
        if os.path.exists(dest): os.remove(dest)

    if vus % 100 == 0:
        pd.DataFrame(infos).to_csv(index_path, index=False)   # sauvegarde progressive
        print(f'  ... {vus} examinées, {ok} frontales gardées (sauvegardé)', flush=True)

pd.DataFrame(infos).to_csv(index_path, index=False)
print('=== TERMINÉ ===', flush=True)
print('Images frontales téléchargées :', ok, flush=True)
print('Index sauvegardé :', index_path, flush=True)
print('Prochaine étape : encodage BiomedCLIP sur GPU (script 02).', flush=True)
