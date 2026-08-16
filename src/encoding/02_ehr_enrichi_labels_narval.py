#=====================================================================
# 02_ehr_enrichi_labels_narval.py
# À lancer sur le NŒUD DE CONNEXION de Narval (internet) — SANS GPU.
# Pour les patients qui ont des images, il construit :
#   - un EHR ENRICHI : diagnostics + labos + signes vitaux + médicaments
#   - les ÉTIQUETTES de PLUSIEURS TÂCHES :
#       * détection de maladie (CheXpert, 14 maladies)
#       * prédiction d'évolution (durée de séjour, mortalité, réadmission)
# À lancer APRÈS 01_telecharger_narval.py (qui produit ).
# =====================================================================

import os, subprocess
import pandas as pd, numpy as np

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
RAW  = os.path.join(BASE, 'data/raw')
os.makedirs(RAW, exist_ok=True)

IV      = 'https://physionet.org/files/mimiciv/3.1/'
CXR_JPG = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'

def lire_identifiants():
    with open(os.path.join(BASE, '.physionet')) as f:
        m = f.read().split()
    return m[m.index('login')+1], m[m.index('password')+1]
USER, PWD = lire_identifiants()

def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0: return
    print('  téléchargement :', os.path.basename(dest), '...', flush=True)
    subprocess.run(['wget', '-q', '--user', USER, '--password', PWD, '-O', dest, url])

# ---------- cohorte = patients qui ont des images ----------
idx = pd.read_csv(os.path.join(RAW, 'image_index_jpg.csv'))
cohorte = set(idx['subject_id'].unique())
print('Cohorte (patients avec images) :', len(cohorte), flush=True)

# =====================================================================
# 1) BASE : démographie + diagnostics -> triplets
# =====================================================================
print('=== 1) Démographie + diagnostics ===', flush=True)
for f in ['hosp/patients.csv.gz','hosp/admissions.csv.gz','hosp/diagnoses_icd.csv.gz','hosp/d_icd_diagnoses.csv.gz']:
    download(IV+f, os.path.join(RAW, os.path.basename(f)))
patients=pd.read_csv(os.path.join(RAW,'patients.csv.gz'))
admissions=pd.read_csv(os.path.join(RAW,'admissions.csv.gz'))
diagnoses=pd.read_csv(os.path.join(RAW,'diagnoses_icd.csv.gz'))
d_icd=pd.read_csv(os.path.join(RAW,'d_icd_diagnoses.csv.gz'))

diag=diagnoses.merge(d_icd,on=['icd_code','icd_version'],how='left').dropna(subset=['long_title'])
sexe={'M':'male','F':'female'}
base_text={}
pp=patients[patients.subject_id.isin(cohorte)]
for _,r in pp.iterrows():
    subj=r['subject_id']; g=sexe.get(r['gender'],'unknown')
    parts=[f"patient gender {g} .", f"patient age {int(r['anchor_age'])} ."]
    for t in diag[diag.subject_id==subj]['long_title'].unique():
        parts.append(f"patient has_diagnosis {t} .")
    base_text[subj]=' '.join(parts)
print('Textes de base construits :', len(base_text), flush=True)

# =====================================================================
# 2) LABOS (labevents) -> bas/normal/haut
# =====================================================================
print('=== 2) Analyses de laboratoire (gros fichier) ===', flush=True)
download(IV+'hosp/labevents.csv.gz', os.path.join(RAW,'labevents.csv.gz'))
LABS={50912:'Creatinine',50971:'Potassium',50983:'Sodium',51222:'Hemoglobin',
      51301:'White Blood Cells',51265:'Platelet Count',50931:'Glucose',
      51006:'Urea Nitrogen',50882:'Bicarbonate',50893:'Calcium',50813:'Lactate'}
cols=['subject_id','itemid','valuenum','ref_range_lower','ref_range_upper']
mor=[]
for ch in pd.read_csv(os.path.join(RAW,'labevents.csv.gz'), usecols=cols, chunksize=2_000_000):
    sub=ch[(ch.subject_id.isin(cohorte))&(ch.itemid.isin(LABS))&(ch.valuenum.notna())]
    if len(sub): mor.append(sub)
labs=pd.concat(mor,ignore_index=True) if mor else pd.DataFrame(columns=cols)
labs=labs.dropna(subset=['ref_range_lower','ref_range_upper'])
def statut(r):
    if r.valuenum<r.ref_range_lower: return 'low'
    if r.valuenum>r.ref_range_upper: return 'high'
    return 'normal'
lab_text={}
if len(labs):
    labs['statut']=labs.apply(statut,axis=1); labs['nom']=labs['itemid'].map(LABS)
    agg=labs.groupby(['subject_id','nom'])['statut'].agg(lambda s:s.value_counts().index[0]).reset_index()
    for subj,g in agg.groupby('subject_id'):
        lab_text[subj]=' '.join(f'patient lab {r.nom} {r.statut} .' for _,r in g.iterrows())
print('Patients avec labos :', len(lab_text), flush=True)

# =====================================================================
# 3) MÉDICAMENTS (prescriptions)
# =====================================================================
print('=== 3) Médicaments ===', flush=True)
download(IV+'hosp/prescriptions.csv.gz', os.path.join(RAW,'prescriptions.csv.gz'))
mor=[]
for ch in pd.read_csv(os.path.join(RAW,'prescriptions.csv.gz'), usecols=['subject_id','drug'], chunksize=2_000_000):
    sub=ch[ch.subject_id.isin(cohorte)].dropna(subset=['drug'])
    if len(sub): mor.append(sub)
meds=pd.concat(mor,ignore_index=True) if mor else pd.DataFrame(columns=['subject_id','drug'])
med_text={}
for subj,g in meds.groupby('subject_id'):
    top=g['drug'].value_counts().head(15).index.tolist()
    med_text[subj]=' '.join(f'patient medication {d} .' for d in top)
print('Patients avec médicaments :', len(med_text), flush=True)

# =====================================================================
# 4) SIGNES VITAUX (chartevents, le plus gros)
# =====================================================================
print('=== 4) Signes vitaux (très gros fichier) ===', flush=True)
download(IV+'icu/chartevents.csv.gz', os.path.join(RAW,'chartevents.csv.gz'))
VIT={220045:('Heart Rate',20,250),220210:('Respiratory Rate',3,60),
     220277:('SpO2',40,100),223761:('Temperature',90,110),
     220179:('Systolic BP',40,300),220180:('Diastolic BP',20,200)}
ids=set(VIT); mor=[]
for ch in pd.read_csv(os.path.join(RAW,'chartevents.csv.gz'), usecols=['subject_id','itemid','valuenum'], chunksize=3_000_000):
    sub=ch[(ch.subject_id.isin(cohorte))&(ch.itemid.isin(ids))&(ch.valuenum.notna())]
    if len(sub): mor.append(sub)
vit=pd.concat(mor,ignore_index=True) if mor else pd.DataFrame(columns=['subject_id','itemid','valuenum'])
vital_text={}
if len(vit):
    def ok(r):
        _,lo,hi=VIT[r.itemid]; return lo<=r.valuenum<=hi
    vit=vit[vit.apply(ok,axis=1)]; vit['nom']=vit['itemid'].map(lambda i:VIT[i][0])
    agg=vit.groupby(['subject_id','nom'])['valuenum'].median().round(0).reset_index()
    for subj,g in agg.groupby('subject_id'):
        vital_text[subj]=' '.join(f'patient vital {r.nom} {int(r.valuenum)} .' for _,r in g.iterrows())
print('Patients avec signes vitaux :', len(vital_text), flush=True)

# =====================================================================
# 5) COMBINER -> EHR enrichi
# =====================================================================
lignes=[]
for subj in cohorte:
    parts=[base_text.get(subj,'')]
    for d in (lab_text,med_text,vital_text):
        if subj in d: parts.append(d[subj])
    lignes.append({'subject_id':subj,'texte':' '.join(p for p in parts if p).strip()})
enrichi=pd.DataFrame(lignes)
enrichi.to_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'),index=False)
print('EHR enrichi sauvegardé :', len(enrichi), 'patients', flush=True)

# =====================================================================
# 6) ÉTIQUETTES — TÂCHE 1 : détection de maladie (CheXpert, 14 maladies)
# =====================================================================
print('=== 6) Étiquettes détection (CheXpert) ===', flush=True)
download(CXR_JPG+'mimic-cxr-2.0.0-chexpert.csv.gz', os.path.join(RAW,'chexpert.csv.gz'))
chex=pd.read_csv(os.path.join(RAW,'chexpert.csv.gz'))
chex[chex.subject_id.isin(cohorte)].to_csv(os.path.join(RAW,'labels_detection.csv'),index=False)
print('labels_detection.csv (14 maladies par étude)', flush=True)

# =====================================================================
# 7) ÉTIQUETTES — TÂCHE 2 : prédiction d'évolution
#    durée de séjour, mortalité hospitalière, réadmission <30j
# =====================================================================
print('=== 7) Étiquettes évolution (admissions) ===', flush=True)
adm=admissions[admissions.subject_id.isin(cohorte)].copy()
adm['admittime']=pd.to_datetime(adm['admittime']); adm['dischtime']=pd.to_datetime(adm['dischtime'])
adm['duree_sejour_jours']=(adm['dischtime']-adm['admittime']).dt.total_seconds()/86400.0
adm=adm.sort_values(['subject_id','admittime'])
adm['prochaine_admission']=adm.groupby('subject_id')['admittime'].shift(-1)
adm['readmission_30j']=((adm['prochaine_admission']-adm['dischtime']).dt.total_seconds()/86400.0 <= 30).astype('Int64')
out=adm[['subject_id','hadm_id','duree_sejour_jours','hospital_expire_flag','readmission_30j']]
out.to_csv(os.path.join(RAW,'labels_outcome.csv'),index=False)
print('labels_outcome.csv (durée séjour, mortalité, réadmission)', flush=True)

print('=== TERMINÉ ===', flush=True)
print('Fichiers prêts : cohort_ehr_text_enriched.csv, labels_detection.csv, labels_outcome.csv', flush=True)
print('Prochaine étape : encodage images sur GPU (script 03) + entraînement multi-tâches.', flush=True)
