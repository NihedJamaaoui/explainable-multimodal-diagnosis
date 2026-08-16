#!/usr/bin/env python3
# =====================================================================
# 10_selection_grande_cohorte.py
# GRANDE COHORTE — a lancer sur le NOEUD DE CONNEXION (internet), sans GPU.
#
# Nouveaute cle : la cohorte part de MIMIC-CXR (patients QUI ONT une radio)
# puis on croise avec MIMIC-IV -> rendement ~100% au lieu de ~20%.
# On garde les N etudes les plus recentes par patient (standard MedFuse/DrFuse).
#
# Produit : data/raw/cohort_patients.csv  et  data/raw/image_index_jpg.csv
#           + les images dans data/jpg/
# =====================================================================

import os, subprocess, csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
RAW  = os.path.join(BASE, 'data/raw')
JPG  = os.path.join(BASE, 'data/jpg')
os.makedirs(RAW, exist_ok=True)
os.makedirs(JPG, exist_ok=True)

# ---------------- PARAMETRES ----------------
N_PATIENTS        = 12000   # patients vises (qui ont radio ET dossier)
MAX_ETUDES_PATIENT = 2      # nb d etudes les plus recentes gardees par patient
N_PARALLEL        = 16      # telechargements simultanes
VUES_FRONTALES    = ['PA', 'AP']

IV      = 'https://physionet.org/files/mimiciv/3.1/'
CXR_JPG = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'

def lire_identifiants():
    with open(os.path.join(BASE, '.physionet')) as f:
        m = f.read().split()
    return m[m.index('login')+1], m[m.index('password')+1]
USER, PWD = lire_identifiants()

def wget(url, dest, essais=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for i in range(essais):
        subprocess.run(['wget', '-q', '-c', '--timeout=60', '--tries=2',
                        '--user', USER, '--password', PWD, '-O', dest, url])
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
    if os.path.exists(dest):
        os.remove(dest)
    return False

# =====================================================================
# 1) Tables necessaires
# =====================================================================
print('=== 1) Tables ===', flush=True)
for f in ['hosp/patients.csv.gz', 'hosp/diagnoses_icd.csv.gz', 'hosp/d_icd_diagnoses.csv.gz']:
    dest = os.path.join(RAW, os.path.basename(f))
    print('  ', os.path.basename(f), flush=True)
    wget(IV + f, dest)
print('   cxr-metadata', flush=True)
wget(CXR_JPG + 'mimic-cxr-2.0.0-metadata.csv.gz', os.path.join(RAW, 'cxr-metadata.csv.gz'))

patients  = pd.read_csv(os.path.join(RAW, 'patients.csv.gz'))
diagnoses = pd.read_csv(os.path.join(RAW, 'diagnoses_icd.csv.gz'), usecols=['subject_id'])
meta      = pd.read_csv(os.path.join(RAW, 'cxr-metadata.csv.gz'))

# =====================================================================
# 2) COHORTE = patients qui ont une RADIO **et** un DOSSIER
#    (c est ce croisement qui evite de perdre 80% des patients)
# =====================================================================
print('=== 2) Croisement CXR + MIMIC-IV ===', flush=True)
meta_front = meta[meta['ViewPosition'].isin(VUES_FRONTALES)].copy()
pat_cxr = set(meta_front['subject_id'].unique())
pat_ehr = set(patients[(patients['gender'].isin(['M', 'F'])) &
                       (patients['anchor_age'].notna())]['subject_id'])
pat_diag = set(diagnoses['subject_id'].unique())

cohorte_possible = sorted(pat_cxr & pat_ehr & pat_diag)
print('  patients avec radio frontale :', len(pat_cxr), flush=True)
print('  patients avec dossier + diagnostics :', len(pat_ehr & pat_diag), flush=True)
print('  >>> INTERSECTION (utilisable) :', len(cohorte_possible), flush=True)

cohorte = set(cohorte_possible[:N_PATIENTS])
print('  cohorte retenue :', len(cohorte), flush=True)
pd.DataFrame({'subject_id': sorted(cohorte)}).to_csv(os.path.join(RAW, 'cohort_patients.csv'), index=False)

# =====================================================================
# 3) Selection des etudes les plus RECENTES par patient
# =====================================================================
print('=== 3) Selection des etudes recentes ===', flush=True)
sel = meta_front[meta_front['subject_id'].isin(cohorte)].copy()
sel['StudyDate'] = pd.to_numeric(sel['StudyDate'], errors='coerce').fillna(0)
# une seule image par etude, puis les N etudes les plus recentes du patient
sel = sel.sort_values(['subject_id', 'StudyDate'], ascending=[True, False])
sel = sel.drop_duplicates(subset=['study_id'], keep='first')
sel = sel.groupby('subject_id', group_keys=False).head(MAX_ETUDES_PATIENT)
print('  images a telecharger :', len(sel), flush=True)
print('  moyenne images/patient :', round(len(sel) / max(len(cohorte), 1), 2), flush=True)

# =====================================================================
# 4) Telechargement PARALLELE (reprise automatique)
# =====================================================================
index_path = os.path.join(RAW, 'image_index_jpg.csv')
faits = set()
if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
    faits = set(pd.read_csv(index_path)['dicom_id'])
    print('  deja telechargees :', len(faits), flush=True)

def taches():
    for _, r in sel.iterrows():
        if r['dicom_id'] in faits:
            continue
        chemin = f"p{str(r['subject_id'])[:2]}/p{r['subject_id']}/s{r['study_id']}/{r['dicom_id']}.jpg"
        yield (r['subject_id'], r['study_id'], r['dicom_id'],
               CXR_JPG + 'files/' + chemin,
               os.path.join(JPG, f"{r['dicom_id']}.jpg"),
               r['ViewPosition'])

def traiter(t):
    subj, study, dicom, url, dest, vue = t
    if wget(url, dest, essais=2):
        return {'subject_id': subj, 'study_id': study, 'dicom_id': dicom,
                'jpg_path': dest, 'view': vue}
    return None

liste = list(taches())
print('=== 4) Telechargement de', len(liste), 'images (', N_PARALLEL, 'en parallele ) ===', flush=True)

nouveau = (not os.path.exists(index_path)) or os.path.getsize(index_path) == 0
f = open(index_path, 'a', newline='')
w = csv.DictWriter(f, fieldnames=['subject_id', 'study_id', 'dicom_id', 'jpg_path', 'view'])
if nouveau:
    w.writeheader()

ok = len(faits); n = 0
with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
    futures = [ex.submit(traiter, t) for t in liste]
    for fut in as_completed(futures):
        res = fut.result(); n += 1
        if res:
            w.writerow(res); ok += 1
        if n % 500 == 0:
            f.flush()
            print(f'  ... {n}/{len(liste)} traitees, {ok} images au total', flush=True)
f.close()

print('=== TERMINE ===', flush=True)
print('Images JPG totales :', ok, flush=True)
print('Index :', index_path, flush=True)
print('Prochaine etape : encodage GPU (script 11) puis EHR enrichi.', flush=True)
