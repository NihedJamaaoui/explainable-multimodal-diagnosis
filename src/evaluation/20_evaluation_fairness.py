#!/usr/bin/env python3
# =====================================================================
# 20_evaluation_fairness.py
# EQUITE (FAIRNESS) — ecart de performance demographique.
# A lancer APRES un entrainement qui sauvegarde les predictions test.
#
# Mesure si le modele est aussi performant selon :
#   - le SEXE (homme / femme)
#   - la TRANCHE D AGE (<50, 50-70, >70)
# Metrique : AUROC par sous-groupe + ecart (gap) entre groupes.
# (approche Fairlearn : demographic performance gap)
# =====================================================================

import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
RAW  = os.path.join(BASE, 'data/raw')
RES  = os.path.join(BASE, 'resultats')

# fichier de predictions attendu : resultats/predictions_<tag>.csv
#   colonnes : subject_id, y_true, y_pred
tag = sys.argv[1] if len(sys.argv) > 1 else 'Pleural_Effusion_enrichi'
pred_path = os.path.join(RES, f'predictions_{tag}.csv')
if not os.path.exists(pred_path):
    print('ERREUR : fichier de predictions introuvable :', pred_path)
    print('Astuce : modifie le script d entrainement pour sauvegarder')
    print('  un CSV predictions_<tag>.csv (subject_id, y_true, y_pred).')
    sys.exit(1)

pred = pd.read_csv(pred_path)

# demographie
patients = pd.read_csv(os.path.join(RAW, 'patients.csv.gz'), usecols=['subject_id','gender','anchor_age'])
df = pred.merge(patients, on='subject_id', how='left')

C_A='#1C7293'; C_B='#2E7D5B'; C_RED='#B0413E'

def auroc_sous_groupe(sub):
    if sub['y_true'].nunique() < 2 or len(sub) < 20:
        return None
    return roc_auc_score(sub['y_true'], sub['y_pred'])

resultats = {}

# ---- par SEXE ----
print('=== Equite par SEXE ===', flush=True)
sexe_scores = {}
for g, nom in [('M','Homme'),('F','Femme')]:
    a = auroc_sous_groupe(df[df.gender==g])
    sexe_scores[nom] = a
    print(f'  {nom:6s}: AUROC = {a:.3f}' if a else f'  {nom}: trop peu de donnees', flush=True)
vals = [v for v in sexe_scores.values() if v is not None]
gap_sexe = (max(vals)-min(vals)) if len(vals)==2 else None
if gap_sexe is not None:
    print(f'  >>> Ecart (gap) sexe : {gap_sexe:.3f}', flush=True)
resultats['sexe'] = {'scores':sexe_scores,'gap':gap_sexe}

# ---- par AGE ----
print('=== Equite par TRANCHE D AGE ===', flush=True)
def tranche(a):
    if a < 50: return '<50'
    if a <= 70: return '50-70'
    return '>70'
df['tranche'] = df['anchor_age'].apply(tranche)
age_scores = {}
for t in ['<50','50-70','>70']:
    a = auroc_sous_groupe(df[df.tranche==t])
    age_scores[t] = a
    print(f'  {t:6s}: AUROC = {a:.3f}' if a else f'  {t}: trop peu de donnees', flush=True)
vals2 = [v for v in age_scores.values() if v is not None]
gap_age = (max(vals2)-min(vals2)) if len(vals2)>=2 else None
if gap_age is not None:
    print(f'  >>> Ecart (gap) age : {gap_age:.3f}', flush=True)
resultats['age'] = {'scores':age_scores,'gap':gap_age}

# ---- graphe ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
s1={k:v for k,v in sexe_scores.items() if v is not None}
ax1.bar(list(s1.keys()), list(s1.values()), color=[C_A,C_B])
ax1.set_ylim(0,1); ax1.set_ylabel('AUROC'); ax1.set_title(f'Equite par sexe (gap={gap_sexe:.3f})' if gap_sexe else 'Equite par sexe')
ax1.grid(axis='y',alpha=0.3)
s2={k:v for k,v in age_scores.items() if v is not None}
ax2.bar(list(s2.keys()), list(s2.values()), color=[C_A,C_B,C_RED][:len(s2)])
ax2.set_ylim(0,1); ax2.set_title(f'Equite par age (gap={gap_age:.3f})' if gap_age else 'Equite par age')
ax2.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(RES, f'fairness_{tag}.png'),dpi=150,bbox_inches='tight')

with open(os.path.join(RES, f'fairness_{tag}.json'),'w') as f:
    json.dump(resultats, f, indent=2)

print('=== TERMINE ===', flush=True)
print('Interpretation : un gap FAIBLE (<0.05) = modele equitable entre groupes.', flush=True)
print('Fichiers : fairness_'+tag+'.png, fairness_'+tag+'.json', flush=True)
