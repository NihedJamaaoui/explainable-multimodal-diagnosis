#!/usr/bin/env python3
# =====================================================================
# 40_base_connaissance.py
# NIVEAU 1 — Generation de rapports : construction de la base RAG.
#
# But
# ---
# Construire une base de connaissance clinique pour les 5 pathologies,
# combinant DEUX sources :
#   (A) des descriptions de reference redigees (fiables, controlees)
#   (B) des phrases reelles extraites des rapports MIMIC-CXR de la cohorte
# On encode tout en embeddings. La recherche se fait en NumPy (cosinus),
# sans FAISS : la base est petite (~quelques centaines de passages), donc
# une simple multiplication matricielle est aussi rapide et n a AUCUNE
# dependance a compiler (faiss ne s installe pas sur Narval, swig manque).
#
# Sorties (dans data/rag/) :
#   rag_passages.json   : les passages (texte + metadonnees)
#   rag_embeddings.npy  : la matrice des embeddings (pour la recherche)
#
# A lancer sur le noeud de CONNEXION (telecharge un petit modele).
# Pre-requis : sentence-transformers  (faiss N EST PAS necessaire)
# =====================================================================

import os, re, json
import numpy as np, pandas as pd

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
RAW=os.path.join(BASE,'data/raw')
RAG=os.path.join(BASE,'data/rag'); os.makedirs(RAG,exist_ok=True)

MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']

# =====================================================================
# (A) DESCRIPTIONS DE REFERENCE — fiables, controlees
# =====================================================================
REFERENCE={
'Pleural Effusion':[
 "Pleural effusion is an abnormal accumulation of fluid in the pleural space between the lung and the chest wall.",
 "On a frontal chest radiograph, a pleural effusion typically blunts the costophrenic angle and forms a meniscus at the lung base.",
 "Large effusions may opacify the lower hemithorax and shift the mediastinum to the opposite side.",
 "The finding predominates in the lower lung zones and the costophrenic angles.",
],
'Edema':[
 "Pulmonary edema is the accumulation of fluid in the lung interstitium and alveoli, often of cardiogenic origin.",
 "Radiographic signs include perihilar haziness, Kerley B lines, peribronchial cuffing, and a bilateral perihilar bat-wing pattern.",
 "Edema is typically diffuse and bilateral, predominating in the mid and lower lung zones.",
 "It is frequently associated with cardiomegaly and pleural effusion in heart failure.",
],
'Cardiomegaly':[
 "Cardiomegaly denotes an enlarged cardiac silhouette on the chest radiograph.",
 "It is defined on a PA view by a cardiothoracic ratio greater than 0.5.",
 "The enlarged cardiac silhouette occupies the central and lower mediastinum.",
 "Cardiomegaly is a common sign of chronic heart failure and may accompany pulmonary edema.",
],
'Atelectasis':[
 "Atelectasis is a partial or complete collapse of lung tissue with loss of volume.",
 "Radiographic signs include increased opacity, displacement of fissures, and crowding of vessels.",
 "It commonly appears in the mid and lower lung zones, often as linear or band-like opacities.",
 "Atelectasis may be secondary to bronchial obstruction, compression, or post-operative hypoventilation.",
],
'Pneumonia':[
 "Pneumonia is an infection of the lung parenchyma causing inflammatory consolidation.",
 "On the radiograph it appears as an air-space opacity or consolidation, sometimes with air bronchograms.",
 "Consolidation may be lobar or patchy and predominates in the mid and lower lung zones.",
 "Pneumonia can be difficult to distinguish radiographically from atelectasis or early edema.",
],
}

# =====================================================================
# (B) PHRASES REELLES des rapports MIMIC-CXR de la cohorte
# =====================================================================
MOTS_CLES={
'Pleural Effusion':['effusion','pleural fluid','costophrenic'],
'Edema':['edema','pulmonary edema','vascular congestion','interstitial'],
'Cardiomegaly':['cardiomegaly','cardiac silhouette','heart size','cardiac enlargement'],
'Atelectasis':['atelectasis','atelectatic','volume loss','collapse'],
'Pneumonia':['pneumonia','consolidation','infection','air bronchogram'],
}

REPORTS_DIR=os.path.join(RAW,'mimic-cxr-reports')

def lire_rapport(study_id, subject_id):
    """Lit le fichier .txt d un rapport MIMIC-CXR et renvoie FINDINGS+IMPRESSION."""
    sid=str(subject_id)
    chemin=os.path.join(REPORTS_DIR,'files',f'p{sid[:2]}',f'p{sid}',f's{study_id}.txt')
    if not os.path.exists(chemin): return None
    try:
        txt=open(chemin,encoding='utf-8',errors='ignore').read()
    except Exception:
        return None
    # garder surtout FINDINGS et IMPRESSION (le contenu clinique utile)
    bloc=''
    for cle in ('FINDINGS:','IMPRESSION:'):
        if cle in txt:
            bloc+=' '+txt.split(cle,1)[1].split('\n \n')[0]
    return (bloc if bloc else txt).replace('\n',' ').strip()

def charger_rapports():
    """Parcourt les study_id de la cohorte et lit leurs rapports .txt."""
    if not os.path.isdir(REPORTS_DIR):
        return None, None
    # index cohorte : study_id + subject_id
    idx_path=os.path.join(BASE,'data/processed/image_index_clean.csv')
    if not os.path.exists(idx_path):
        idx_path=os.path.join(RAW,'image_index_jpg.csv')
    idx=pd.read_csv(idx_path)
    paires=idx[['study_id','subject_id']].drop_duplicates()
    print(f'Rapports a lire (cohorte) : {len(paires)} etudes', flush=True)
    lignes=[]
    for _,r in paires.iterrows():
        t=lire_rapport(int(r['study_id']),int(r['subject_id']))
        if t: lignes.append(t)
    if not lignes:
        return None, None
    print(f'Rapports lus avec succes : {len(lignes)}', flush=True)
    return pd.DataFrame({'report':lignes}), 'report'

def phrases_pertinentes(texte,cles,maxn=3):
    if not isinstance(texte,str): return []
    phrases=re.split(r'(?<=[.])\s+', texte)
    out=[]
    for p in phrases:
        pl=p.lower().strip()
        if 8<len(pl)<300 and any(k in pl for k in cles):
            out.append(p.strip())
        if len(out)>=maxn: break
    return out

# =====================================================================
# Construction des passages
# =====================================================================
passages=[]
for m in MALADIES:
    for t in REFERENCE[m]:
        passages.append({'maladie':m,'source':'reference','texte':t})

df,col=charger_rapports()
if df is not None:
    vus=set()
    for _,r in df.iterrows():
        txt=r[col]
        for m in MALADIES:
            for ph in phrases_pertinentes(txt,MOTS_CLES[m]):
                cle=(m,ph.lower()[:60])
                if cle in vus: continue
                vus.add(cle)
                passages.append({'maladie':m,'source':'mimic','texte':ph})
        if len(passages)>800: break
    print(f'Phrases MIMIC ajoutees : {sum(1 for p in passages if p["source"]=="mimic")}', flush=True)
else:
    print('Aucun CSV de rapports trouve — base construite sur les references seules.', flush=True)

print(f'\nTotal passages : {len(passages)}', flush=True)
for m in MALADIES:
    n=sum(1 for p in passages if p['maladie']==m)
    print(f'  {m:18s}: {n} passages', flush=True)

# =====================================================================
# Embeddings (sentence-transformers) — SANS faiss
# =====================================================================
print('\nChargement du modele d embeddings...', flush=True)
from sentence_transformers import SentenceTransformer
enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
textes=[p['texte'] for p in passages]
emb=enc.encode(textes,convert_to_numpy=True,show_progress_bar=True,normalize_embeddings=True)
emb=emb.astype('float32')
print('Embeddings :',emb.shape, flush=True)

# =====================================================================
# Sauvegardes
# =====================================================================
json.dump(passages,open(os.path.join(RAG,'rag_passages.json'),'w'),indent=2,ensure_ascii=False)
np.save(os.path.join(RAG,'rag_embeddings.npy'),emb)

print('\n=== BASE RAG CONSTRUITE ===', flush=True)
print(f'  {os.path.join(RAG,"rag_passages.json")}', flush=True)
print(f'  {os.path.join(RAG,"rag_embeddings.npy")}', flush=True)
print(f'  {len(passages)} passages indexes.', flush=True)

# =====================================================================
# Recherche NumPy (cosinus) — remplace FAISS
# =====================================================================
def rechercher(requete,k=3):
    """Retourne les k passages les plus proches (cosinus)."""
    q=enc.encode([requete],convert_to_numpy=True,normalize_embeddings=True).astype('float32')
    scores=(emb@q[0])                      # produit scalaire = cosinus (vecteurs normalises)
    idx=np.argsort(-scores)[:k]
    return [(passages[i],float(scores[i])) for i in idx]

print('\n--- Test : recherche pour « fluid at the lung base » ---', flush=True)
for rang,(p,s) in enumerate(rechercher('fluid at the lung base',3),1):
    print(f'  {rang}. [{p["maladie"]} / {p["source"]}] {p["texte"][:80]}  (score {s:.2f})', flush=True)

print('\n=== TERMINE ===', flush=True)
print('La recherche se fait en NumPy (pas besoin de faiss).', flush=True)
print('Prochaine etape : precharger le LLM (script 41).', flush=True)
