#!/usr/bin/env python3
# =====================================================================
# 31b_corriger_dimensions.py
# CORRECTION DES DIMENSIONS D IMAGE DANS imagenome_boxes.csv
#
# Probleme constate : les colonnes image_width / image_height du fichier
# extrait contiennent tres probablement les dimensions de la BOITE et non
# celles de la RADIOGRAPHIE. La conversion vers la grille 14x14 echoue
# alors pour la quasi-totalite des images.
#
# Ce script procede en trois temps :
#   1. DIAGNOSTIC  : compare les dimensions stockees aux dimensions reelles
#                    lues dans les fichiers JPG, sur un echantillon.
#   2. CORRECTION  : reecrit les colonnes avec les vraies dimensions.
#   3. SIMULATION  : construit les masques et rapporte, pathologie par
#                    pathologie, combien d images seraient exploitables et
#                    quelle part de la grille serait couverte.
#
# A lancer sur le noeud de connexion (aucun GPU necessaire).
# Sortie : data/raw/imagenome_boxes_corrige.csv
# =====================================================================

import os, sys
import numpy as np, pandas as pd
from PIL import Image

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
RAW=os.path.join(BASE,'data/raw'); PROC=os.path.join(BASE,'data/processed')
JPG=os.path.join(BASE,'data/jpg')
ENTREE=os.path.join(RAW,'imagenome_boxes.csv')
SORTIE=os.path.join(RAW,'imagenome_boxes_corrige.csv')

MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
NOMS={'Pleural Effusion':'Épanchement','Edema':'Œdème','Cardiomegaly':'Cardiomégalie',
      'Atelectasis':'Atélectasie','Pneumonia':'Pneumonie'}

# Correspondance pathologie -> regions anatomiques.
# Les pathologies diffuses sont volontairement restreintes aux zones
# moyennes et inferieures plutot qu a l ensemble des champs pulmonaires :
# un masque couvrant presque tout le thorax ne contraint plus rien.
ZONES_CIBLEES=['right mid lung zone','right lower lung zone',
               'left mid lung zone','left lower lung zone',
               'right hilar structures','left hilar structures']
REGIONS={
 'Pleural Effusion':['left costophrenic angle','right costophrenic angle',
                     'left lower lung zone','right lower lung zone',
                     'left hemidiaphragm','right hemidiaphragm'],
 'Edema':ZONES_CIBLEES,
 'Cardiomegaly':['left cardiac silhouette','right cardiac silhouette','cardiac silhouette'],
 'Atelectasis':ZONES_CIBLEES,
 'Pneumonia':ZONES_CIBLEES,
}

if not os.path.exists(ENTREE):
    print('ERREUR : fichier introuvable ->',ENTREE, flush=True); sys.exit(1)

boxes=pd.read_csv(ENTREE)
boxes['dicom_id']=boxes['dicom_id'].astype(str)
boxes['region']=boxes['region'].astype(str).str.strip().str.lower()
print('Boites lues :',len(boxes),'|',boxes['dicom_id'].nunique(),'images', flush=True)

# =====================================================================
# 1) DIAGNOSTIC
# =====================================================================
print('\n=== 1) DIAGNOSTIC DES DIMENSIONS ===', flush=True)

def taille_jpg(dic):
    chemin=os.path.join(JPG,f'{dic}.jpg')
    if not os.path.exists(chemin): return None
    try:
        with Image.open(chemin) as im:   # PIL ne lit que l en-tete pour .size
            return im.size               # (largeur, hauteur)
    except Exception:
        return None

echantillon=boxes['dicom_id'].drop_duplicates().head(5).tolist()
print(f"{'dicom_id':>14} | {'stockee':>13} | {'reelle (JPG)':>13} | boite (x1,y1,x2,y2)", flush=True)
for dic in echantillon:
    g=boxes[boxes.dicom_id==dic].iloc[0]
    reelle=taille_jpg(dic)
    stockee=f"{g['image_width']}x{g['image_height']}"
    r=f"{reelle[0]}x{reelle[1]}" if reelle else 'introuvable'
    print(f"{dic[:14]:>14} | {stockee:>13} | {r:>13} | "
          f"({g['x1']:.0f},{g['y1']:.0f},{g['x2']:.0f},{g['y2']:.0f})", flush=True)

print('\nLecture attendue : si la colonne « stockee » est nettement plus petite', flush=True)
print('que « reelle » et de l ordre de grandeur de la boite, le diagnostic est', flush=True)
print('confirme — les dimensions enregistrees sont celles de la boite.', flush=True)

# =====================================================================
# 2) CORRECTION
# =====================================================================
print('\n=== 2) CORRECTION DEPUIS LES FICHIERS JPG ===', flush=True)
tailles={}; manquantes=0
for k,dic in enumerate(boxes['dicom_id'].unique()):
    t=taille_jpg(dic)
    if t is None: manquantes+=1
    else: tailles[dic]=t
    if (k+1)%4000==0: print(f'  ... {k+1} images inspectees', flush=True)

print(f'Dimensions recuperees pour {len(tailles)} images ({manquantes} JPG introuvables)', flush=True)
avant=len(boxes)
boxes=boxes[boxes['dicom_id'].isin(tailles)].copy()
boxes['image_width'] =boxes['dicom_id'].map(lambda d: tailles[d][0])
boxes['image_height']=boxes['dicom_id'].map(lambda d: tailles[d][1])
print(f'Boites conservees : {len(boxes)} / {avant}', flush=True)

# controle de coherence : les coordonnees doivent tenir dans l image
hors=((boxes['x2']>boxes['image_width']*1.02)|(boxes['y2']>boxes['image_height']*1.02)).sum()
print(f'Boites depassant le cadre de l image : {hors}'
      f'  ({100*hors/max(len(boxes),1):.1f}%)', flush=True)
if hors>0.2*len(boxes):
    print('ATTENTION : proportion elevee. Les coordonnees ne sont peut-etre pas', flush=True)
    print('exprimees dans le repere de l image d origine.', flush=True)

boxes.to_csv(SORTIE,index=False)
print('Fichier corrige ecrit :',SORTIE, flush=True)

# =====================================================================
# 3) SIMULATION DES MASQUES
# =====================================================================
print('\n=== 3) SIMULATION DES MASQUES ===', flush=True)

index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)
etiquettes={str(r['dicom_id']):[int(r[m]) for m in MALADIES] for _,r in index.iterrows()}

dispo=set(boxes['region'].unique())
REG_OK={}
print('\nCorrespondance pathologie -> regions :', flush=True)
for mal,regs in REGIONS.items():
    presentes=[r.lower() for r in regs if r.lower() in dispo]
    REG_OK[mal]=set(presentes)
    print(f'  {NOMS[mal]:14s}: {len(presentes)}/{len(regs)} regions presentes', flush=True)
    absentes=[r for r in regs if r.lower() not in dispo]
    if absentes: print(f'      absentes : {absentes}', flush=True)

def boite_vers_grille(x1,y1,x2,y2,W,H):
    if not W or not H or W<=0 or H<=0: return None
    s=224.0/min(W,H); nW,nH=W*s,H*s
    dx,dy=(nW-224.0)/2.0,(nH-224.0)/2.0
    a=[x1*s-dx, y1*s-dy, x2*s-dx, y2*s-dy]
    a=[max(0.0,min(224.0,v)) for v in a]
    if a[2]-a[0]<4 or a[3]-a[1]<4: return None
    return [a[0]/224.0,a[1]/224.0,a[2]/224.0,a[3]/224.0]

def poser(m,b):
    c1=int(np.floor(b[0]*14)); c2=int(np.ceil(b[2]*14))
    r1=int(np.floor(b[1]*14)); r2=int(np.ceil(b[3]*14))
    m[max(0,r1):min(14,r2), max(0,c1):min(14,c2)]=1.0

utiles=set().union(*REG_OK.values())
pertinentes=boxes[boxes['region'].isin(utiles)]
print(f'\nBoites pertinentes : {len(pertinentes)} sur {len(boxes)}', flush=True)

ok=0; echec_geo=0; sans_region=0
couvertures=[]; par_maladie={m:[0,[]] for m in MALADIES}
for dic,g in pertinentes.groupby('dicom_id'):
    y=etiquettes.get(dic)
    if y is None: continue
    positives=[MALADIES[k] for k in range(len(MALADIES)) if y[k]==1]
    if not positives: continue
    regs=set().union(*[REG_OK[m] for m in positives])
    sous=g[g['region'].isin(regs)]
    if len(sous)==0:
        sans_region+=1; continue
    m14=np.zeros((14,14),dtype=np.float32); pose=False
    for _,r in sous.iterrows():
        b=boite_vers_grille(float(r['x1']),float(r['y1']),float(r['x2']),float(r['y2']),
                            float(r['image_width']),float(r['image_height']))
        if b is None: continue
        poser(m14,b); pose=True
    if pose:
        ok+=1; c=float(m14.mean()); couvertures.append(c)
        for m in positives:
            par_maladie[m][0]+=1; par_maladie[m][1].append(c)
    else:
        echec_geo+=1

print('\n--- Resultat de la simulation ---', flush=True)
print(f'  Images avec masque exploitable  : {ok}', flush=True)
print(f'  Echec de conversion geometrique : {echec_geo}', flush=True)
print(f'  Aucune region pertinente        : {sans_region}', flush=True)
if couvertures:
    print(f'  Couverture moyenne de la grille : {np.mean(couvertures):.1%}', flush=True)
    print(f'  Couverture mediane              : {np.median(couvertures):.1%}', flush=True)

print('\n--- Par pathologie ---', flush=True)
for m in MALADIES:
    n,cs=par_maladie[m]
    if n: print(f'  {NOMS[m]:14s}: {n:6d} images | couverture {np.mean(cs):.1%}', flush=True)
    else: print(f'  {NOMS[m]:14s}: aucune image', flush=True)

print('\n--- Lecture ---', flush=True)
if ok>3000:
    print('  Effectif suffisant : L_ground peut etre entrainee.', flush=True)
elif ok>500:
    print('  Effectif limite : L_ground aura un effet, mais mesure avec prudence.', flush=True)
else:
    print('  Effectif insuffisant : la conversion echoue encore, ne pas lancer', flush=True)
    print('  l entrainement avant d avoir compris pourquoi.', flush=True)
if couvertures and np.mean(couvertures)>0.65:
    print('  Couverture elevee : le masque contraint peu. Envisager de restreindre', flush=True)
    print('  la correspondance a des regions plus etroites.', flush=True)

print('\n=== TERMINE ===', flush=True)
print('Si la simulation est concluante, pointer 32_lground.py vers', flush=True)
print('imagenome_boxes_corrige.csv puis relancer le job.', flush=True)
